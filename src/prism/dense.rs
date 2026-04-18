//! prism/dense.rs
//! SMPSO-local dense graph + solution + objective + crowding-distance.
//! Bypasses HP-MOCD hash-map-based structures to eliminate lookup overhead
//! that profiling showed dominated runtime (~56% in objective, ~26% in particle update).
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::cmp::Ordering;

pub type DensePartition = Vec<CommunityId>;

/// Dense, cache-friendly view over a Graph. Built once at SMPSO entry.
/// Node indices in `[0, n)` are dense; `nodes[i]` maps back to the original NodeId.
pub struct DenseGraph {
    pub n: usize,
    pub total_edges: usize,
    pub degrees: Vec<u32>,
    pub adj: Vec<Vec<u32>>,
    /// Each undirected edge once, `u < v` (dense indices).
    pub edges: Vec<(u32, u32)>,
    pub nodes: Vec<NodeId>,
    /// TOM / Jaccard weight per `edges[i]`. Parallel array.
    pub edge_w: Vec<f64>,
    /// Σ of incident-edge weights per node.
    pub weighted_deg: Vec<f64>,
    /// Σ of all edge weights.
    pub total_weight: f64,
}

impl DenseGraph {
    pub fn from_graph(g: &Graph) -> Self {
        let nodes: Vec<NodeId> = g.nodes_vec().clone();
        let n = nodes.len();
        let mut node_to_idx: FxHashMap<NodeId, u32> =
            FxHashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for (i, &nid) in nodes.iter().enumerate() {
            node_to_idx.insert(nid, i as u32);
        }

        let mut degrees: Vec<u32> = Vec::with_capacity(n);
        let mut adj: Vec<Vec<u32>> = Vec::with_capacity(n);
        for &nid in &nodes {
            let neighbors = g.neighbors(&nid);
            let mut dense_nb: Vec<u32> = Vec::with_capacity(neighbors.len());
            for &nb in neighbors {
                if let Some(&j) = node_to_idx.get(&nb) {
                    dense_nb.push(j);
                }
            }
            dense_nb.sort_unstable();
            degrees.push(dense_nb.len() as u32);
            adj.push(dense_nb);
        }

        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(g.num_edges());
        for i in 0..n {
            let iu = i as u32;
            for &j in &adj[i] {
                if j > iu {
                    edges.push((iu, j));
                }
            }
        }

        // TOM / Jaccard reweighting: w(u,v) = (|N+(u) ∩ N+(v)|) / (|N+(u) ∪ N+(v)|)
        // where N+(x) = N(x) ∪ {x}. Boosts edges shared between densely-connected pairs,
        // suppresses noisy inter-community edges that lack triangles.
        let mut edge_w: Vec<f64> = Vec::with_capacity(edges.len());
        let mut weighted_deg: Vec<f64> = vec![0.0; n];
        for &(u, v) in &edges {
            let a = &adj[u as usize];
            let b = &adj[v as usize];
            // Jaccard on augmented neighborhoods (include u in N(v), v in N(u)).
            // Count intersection via two-pointer on sorted lists, then add u,v adjustments.
            let mut i = 0usize;
            let mut j = 0usize;
            let mut inter = 0u32;
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    Ordering::Equal => {
                        inter += 1;
                        i += 1;
                        j += 1;
                    }
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                }
            }
            // Augmented: u ∈ N+(v) (adj[v] contains u), v ∈ N+(u). So intersection
            // already counts u and v if they appear in each other's adj (which they do for edges).
            // Plus each self-augmentation: add 2 to inter (u∈N+(u), v∈N+(v)).
            // Simpler: use the classic form |N+(u)∩N+(v)|/min(|N+(u)|,|N+(v)|) variant,
            // which is numerically robust — but stick to Jaccard for null-model-alignment.
            let aug_inter = (inter + 2) as f64; // +u, +v self-inclusions
            let union = (a.len() + b.len()) as f64 - inter as f64; // |N∪N'|
            let aug_union = union + 2.0 - 2.0; // {u}∪{v} add 2 but overlap with adj → net +0; keep simple
            // Guard: union min 1
            let denom = aug_union.max(1.0);
            // Small ε so edge never drops to zero (preserves graph topology).
            let w = (aug_inter / denom).max(1e-3);
            edge_w.push(w);
            weighted_deg[u as usize] += w;
            weighted_deg[v as usize] += w;
        }
        let total_weight: f64 = edge_w.iter().sum();

        DenseGraph {
            n,
            total_edges: g.num_edges(),
            degrees,
            adj,
            edges,
            nodes,
            edge_w,
            weighted_deg,
            total_weight,
        }
    }

    pub fn sparse_partition(&self, dense: &[CommunityId]) -> Partition {
        let mut p = Partition::default();
        p.reserve(self.n);
        for (i, &c) in dense.iter().enumerate() {
            p.insert(self.nodes[i], c);
        }
        p
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub partition: DensePartition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Solution {
    pub fn new(partition: DensePartition) -> Self {
        Solution {
            partition,
            objectives: Vec::new(),
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    #[inline(always)]
    pub fn dominates(&self, other: &Solution) -> bool {
        let mut at_least_one_better = false;
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }
}

/// Rust-objectives path: compute (intra, inter) over a dense partition.
/// Mirrors `operators::objective::calculate_objectives` minus all HashMap lookups.
#[allow(dead_code)]
pub fn evaluate_rust(dg: &DenseGraph, p: &[CommunityId]) -> [f64; 2] {
    let total_edges = dg.total_edges as f64;
    if total_edges == 0.0 {
        return [0.0, 0.0];
    }

    let mut intra_edges: u64 = 0;
    for &(u, v) in &dg.edges {
        if p[u as usize] == p[v as usize] {
            intra_edges += 1;
        }
    }

    let mut comm_deg: FxHashMap<CommunityId, u64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    for i in 0..dg.n {
        *comm_deg.entry(p[i]).or_insert(0) += dg.degrees[i] as u64;
    }

    let intra = 1.0 - (intra_edges as f64) / total_edges;
    let total2 = 2.0 * total_edges;
    let mut inter = 0.0;
    for &d in comm_deg.values() {
        let f = (d as f64) / total2;
        inter += f * f;
    }
    [intra, inter]
}

/// Per-rank crowding distance over SMPSO `Solution`s — objectives only.
/// All archive members share rank = 1, so this is a single-group sweep.
pub fn crowding_distance(members: &mut [Solution]) {
    let n = members.len();
    if n == 0 {
        return;
    }
    for m in members.iter_mut() {
        m.crowding_distance = 0.0;
    }
    if n <= 2 {
        for m in members.iter_mut() {
            m.crowding_distance = f64::INFINITY;
        }
        return;
    }

    let n_obj = members[0].objectives.len();
    let mut order: Vec<usize> = (0..n).collect();

    for obj_idx in 0..n_obj {
        order.sort_unstable_by(|&a, &b| {
            members[a].objectives[obj_idx]
                .partial_cmp(&members[b].objectives[obj_idx])
                .unwrap_or(Ordering::Equal)
        });

        let first = order[0];
        let last = order[n - 1];
        members[first].crowding_distance = f64::INFINITY;
        members[last].crowding_distance = f64::INFINITY;

        let obj_min = members[first].objectives[obj_idx];
        let obj_max = members[last].objectives[obj_idx];
        let span = obj_max - obj_min;
        if span.abs() <= f64::EPSILON {
            continue;
        }
        let scale = 1.0 / span;

        for k in 1..n - 1 {
            let i = order[k];
            let prev = members[order[k - 1]].objectives[obj_idx];
            let next = members[order[k + 1]].objectives[obj_idx];
            members[i].crowding_distance += (next - prev) * scale;
        }
    }
}

/// Sum of objectives → modularity-like Q = n_obj - Σ obj (higher = better).
#[inline(always)]
pub fn q_score(s: &Solution) -> f64 {
    (s.objectives.len() as f64) - s.objectives.iter().sum::<f64>()
}

/// Multi-resolution weighted modularity Pareto: [-Q_0.5, -Q_1.0, -Q_2.0].
/// All minimised. Uses TOM/Jaccard-reweighted graph.
///
/// Q_γ = Σ_c [ L_c / m_w  -  γ · (vol_c / 2 m_w)² ]
///
/// Three γ axes span community scales simultaneously: γ=0.5 favours coarse
/// groupings, γ=2.0 penalises large communities so small ground-truth
/// communities survive on the Pareto front. Structurally defeats the classic
/// modularity resolution limit and prevents giant-community collapse at
/// high μ (γ=2.0 dominates any single-community solution).
pub fn evaluate_q_gamma(dg: &DenseGraph, p: &[CommunityId]) -> [f64; 3] {
    if dg.n == 0 || dg.total_weight <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let m2 = 2.0 * dg.total_weight;
    let mut intra: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    let mut wdeg: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    for (i, &c) in p.iter().enumerate() {
        *wdeg.entry(c).or_insert(0.0) += dg.weighted_deg[i];
    }
    for (eidx, &(u, v)) in dg.edges.iter().enumerate() {
        let cu = p[u as usize];
        let cv = p[v as usize];
        if cu == cv {
            *intra.entry(cu).or_insert(0.0) += dg.edge_w[eidx];
        }
    }
    let mut q05 = 0.0;
    let mut q10 = 0.0;
    let mut q20 = 0.0;
    for (c, &vol) in wdeg.iter() {
        let ec = *intra.get(c).unwrap_or(&0.0);
        let l = ec / dg.total_weight;
        let v2 = (vol / m2).powi(2);
        q05 += l - 0.5 * v2;
        q10 += l - v2;
        q20 += l - 2.0 * v2;
    }
    [-q05, -q10, -q20]
}

/// Weighted modularity (-Q) + avg conductance (Q, conductance) pair.
/// Both minimized (we minimize -Q and avg conductance).
/// Uses TOM/Jaccard-reweighted graph to survive high-μ regimes where
/// absolute edge counts become non-informative (inter > intra).
#[allow(dead_code)]
pub fn evaluate_mod_cond(dg: &DenseGraph, p: &[CommunityId]) -> [f64; 2] {
    if dg.n == 0 || dg.total_weight <= 0.0 {
        return [0.0, 0.0];
    }
    let m2 = 2.0 * dg.total_weight;

    // Intra weight + weighted-degree sum per community.
    let mut intra: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    let mut cut: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    let mut wdeg: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);

    for (i, &c) in p.iter().enumerate() {
        *wdeg.entry(c).or_insert(0.0) += dg.weighted_deg[i];
    }
    for (eidx, &(u, v)) in dg.edges.iter().enumerate() {
        let w = dg.edge_w[eidx];
        let cu = p[u as usize];
        let cv = p[v as usize];
        if cu == cv {
            *intra.entry(cu).or_insert(0.0) += w;
        } else {
            *cut.entry(cu).or_insert(0.0) += w;
            *cut.entry(cv).or_insert(0.0) += w;
        }
    }

    // Q_w = Σ_c [ L_c / m_w  -  (vol_c / 2m_w)^2 ]
    let mut q = 0.0;
    let mut cond_sum = 0.0;
    let mut k = 0usize;
    for (c, &vol) in wdeg.iter() {
        let ec = *intra.get(c).unwrap_or(&0.0);
        q += ec / dg.total_weight - (vol / m2).powi(2);
        // Conductance_c = cut_c / min(vol_c, 2m_w - vol_c); guard for isolates.
        let cc = *cut.get(c).unwrap_or(&0.0);
        let denom = vol.min(m2 - vol);
        let cond = if denom > 1e-12 { cc / denom } else { 1.0 };
        cond_sum += cond;
        k += 1;
    }
    let avg_cond = if k > 0 { cond_sum / (k as f64) } else { 1.0 };
    // Minimize both → return negatives where appropriate.
    [-q, avg_cond]
}

/// NRA + RC objectives (Extended SMPSO). Both minimized.
/// f1 = -Σ_c (E_c / |V_c|)  — Negative Ratio Association (densify intras)
/// f2 =  Σ_c (cut_c / |V_c|) — Ratio Cut (sparsen inters)
/// E_c   = intra-community edge count for community c
/// cut_c = edges leaving community c
/// |V_c| = number of nodes in c
#[allow(dead_code)]
pub fn evaluate_nra_rc(dg: &DenseGraph, p: &[CommunityId]) -> [f64; 2] {
    if dg.n == 0 || dg.total_edges == 0 {
        return [0.0, 0.0];
    }
    let mut intra: FxHashMap<CommunityId, u64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    let mut cut: FxHashMap<CommunityId, u64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    let mut size: FxHashMap<CommunityId, u64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);

    for &c in p.iter() {
        *size.entry(c).or_insert(0) += 1;
    }
    for &(u, v) in &dg.edges {
        let cu = p[u as usize];
        let cv = p[v as usize];
        if cu == cv {
            *intra.entry(cu).or_insert(0) += 1;
        } else {
            *cut.entry(cu).or_insert(0) += 1;
            *cut.entry(cv).or_insert(0) += 1;
        }
    }
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    for (c, &sz) in size.iter() {
        if sz == 0 {
            continue;
        }
        let ec = *intra.get(c).unwrap_or(&0) as f64;
        let cc = *cut.get(c).unwrap_or(&0) as f64;
        let sf = sz as f64;
        f1 -= ec / sf;
        f2 += cc / sf;
    }
    [f1, f2]
}

/// Label Propagation seed: each node iteratively adopts majority label of neighbours.
/// Initial labels = dense node index (one community per node). Converges fast.
pub fn lpa_partition(dg: &DenseGraph, iters: usize) -> DensePartition {
    use rand::RngExt;
    let mut rng = rand::rng();
    let mut p: DensePartition = (0..dg.n).map(|i| i as CommunityId).collect();
    let mut order: Vec<usize> = (0..dg.n).collect();
    let mut freq: FxHashMap<CommunityId, u32> =
        FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);
    let mut tied: Vec<CommunityId> = Vec::with_capacity(8);

    let min_k = ((dg.n as f64).sqrt().floor() as usize).max(4);
    for _ in 0..iters {
        // shuffle order
        for i in (1..dg.n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        // guard: stop if unique label count already dropped below sqrt(n)
        {
            let mut seen = rustc_hash::FxHashSet::default();
            for &c in p.iter() {
                seen.insert(c);
                if seen.len() > min_k {
                    break;
                }
            }
            if seen.len() <= min_k {
                break;
            }
        }
        let mut changed = false;
        for &j in &order {
            let nbs = &dg.adj[j];
            if nbs.is_empty() {
                continue;
            }
            freq.clear();
            for &nb in nbs {
                *freq.entry(p[nb as usize]).or_insert(0) += 1;
            }
            let mut best = 0u32;
            for &v in freq.values() {
                if v > best {
                    best = v;
                }
            }
            tied.clear();
            for (&c, &v) in freq.iter() {
                if v == best {
                    tied.push(c);
                }
            }
            let new_c = if tied.len() == 1 {
                tied[0]
            } else {
                tied[rng.random_range(0..tied.len())]
            };
            if new_c != p[j] {
                p[j] = new_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    p
}
