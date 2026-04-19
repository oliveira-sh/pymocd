//! prism/dense.rs
//! Flat-array data structures for the PRISM hot path.
//! All community-keyed state is a `Vec<T>` sized to `n` (CommunityId ∈ [0, n)
//! by construction), eliminating the FxHashMap lookups that dominated runtime
//! in the previous revision (~67% in evaluate_q_gamma alone under flamegraph).
//!
//! Layout:
//!   - CSR adjacency (`adj_starts`/`adj_flat`) — contiguous, single allocation.
//!   - Packed edges (`edges`: `(u, v, w)`) — AoS for the objective edge loop.
//!   - `Scratch` struct of pre-allocated flat buffers owned by each `Particle`;
//!     reset via `fill(0.0)` per call (memset is O(n), dwarfed by the work).
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::cmp::Ordering;

pub type DensePartition = Vec<CommunityId>;

/// Packed edge: (u, v, tom_weight). 16 bytes. AoS for contiguous iteration
/// in the objective kernel.
pub type Edge = (u32, u32, f64);

/// Dense view over a `Graph`. Built once per `envolve()`.
pub struct DenseGraph {
    pub n: usize,
    pub total_edges: usize,
    pub degrees: Vec<u32>,
    /// CSR offsets: neighbors of node `i` live at `adj_flat[adj_starts[i] .. adj_starts[i+1]]`.
    pub adj_starts: Vec<u32>,
    pub adj_flat: Vec<u32>,
    /// Each undirected edge once (u < v), packed with its TOM/Jaccard weight.
    pub edges: Vec<Edge>,
    pub nodes: Vec<NodeId>,
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

        // Pass 1: build per-node sorted neighbor lists to compute degrees and CSR offsets.
        let mut nb_tmp: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut degrees: Vec<u32> = Vec::with_capacity(n);
        let mut total_neighbors: usize = 0;
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
            total_neighbors += dense_nb.len();
            nb_tmp.push(dense_nb);
        }

        // CSR flatten.
        let mut adj_starts: Vec<u32> = Vec::with_capacity(n + 1);
        let mut adj_flat: Vec<u32> = Vec::with_capacity(total_neighbors);
        adj_starts.push(0);
        for nb in &nb_tmp {
            adj_flat.extend_from_slice(nb);
            adj_starts.push(adj_flat.len() as u32);
        }

        // Collect unique edges (u < v).
        let mut edges_uv: Vec<(u32, u32)> = Vec::with_capacity(g.num_edges());
        for i in 0..n {
            let iu = i as u32;
            for &j in &nb_tmp[i] {
                if j > iu {
                    edges_uv.push((iu, j));
                }
            }
        }

        // TOM/Jaccard reweighting on sorted CSR-resident neighbor lists.
        let mut edges: Vec<Edge> = Vec::with_capacity(edges_uv.len());
        let mut weighted_deg: Vec<f64> = vec![0.0; n];
        for &(u, v) in &edges_uv {
            let a = &nb_tmp[u as usize];
            let b = &nb_tmp[v as usize];
            let mut i = 0usize;
            let mut j = 0usize;
            let mut inter: u32 = 0;
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
            let aug_inter = (inter + 2) as f64;
            let union = (a.len() + b.len()) as f64 - inter as f64;
            let w = (aug_inter / union.max(1.0)).max(1e-3);
            edges.push((u, v, w));
            weighted_deg[u as usize] += w;
            weighted_deg[v as usize] += w;
        }
        let total_weight: f64 = edges.iter().map(|e| e.2).sum();

        DenseGraph {
            n,
            total_edges: g.num_edges(),
            degrees,
            adj_starts,
            adj_flat,
            edges,
            nodes,
            weighted_deg,
            total_weight,
        }
    }

    /// CSR neighbor slice for node `i`. Zero-cost; no allocation.
    #[inline(always)]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let s = self.adj_starts[i] as usize;
        let e = self.adj_starts[i + 1] as usize;
        &self.adj_flat[s..e]
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

/// Flat per-particle scratch pad. All vectors are sized `n` and held at
/// all-zero between calls; the consumer is responsible for zeroing its
/// touched slots (or calling `fill(0.0)` to start from a clean slate).
/// Reused across generations so no per-call allocation.
#[derive(Debug)]
pub struct Scratch {
    pub intra: Vec<f64>,
    pub wdeg: Vec<f64>,
    pub freq_f: Vec<f64>,
    pub freq_u: Vec<u32>,
    pub ctd: Vec<f64>,
    pub touched: Vec<u32>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            intra: vec![0.0; n],
            wdeg: vec![0.0; n],
            freq_f: vec![0.0; n],
            freq_u: vec![0; n],
            ctd: vec![0.0; n],
            touched: Vec::with_capacity(n),
        }
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
            objectives: Vec::with_capacity(3),
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    #[inline(always)]
    pub fn dominates(&self, other: &Solution) -> bool {
        let mut better = false;
        for i in 0..self.objectives.len() {
            let a = self.objectives[i];
            let b = other.objectives[i];
            if a > b {
                return false;
            }
            if a < b {
                better = true;
            }
        }
        better
    }
}

/// Per-rank crowding distance. All archive members share rank = 1.
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

#[inline(always)]
pub fn q_score(s: &Solution) -> f64 {
    (s.objectives.len() as f64) - s.objectives.iter().sum::<f64>()
}

/// Multi-resolution weighted modularity: [-Q_0.5, -Q_1.0, -Q_2.0].
/// All-flat version — no hashmap. Uses `Scratch.intra` and `Scratch.wdeg`
/// (size n). Both are reset via `fill(0.0)` on entry to guarantee cleanliness.
///
/// Q_γ = Σ_c [ L_c / m_w  -  γ · (vol_c / 2·m_w)² ]
pub fn evaluate_q_gamma(dg: &DenseGraph, p: &[CommunityId], s: &mut Scratch) -> [f64; 3] {
    let n = dg.n;
    if n == 0 || dg.total_weight <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    debug_assert_eq!(p.len(), n);
    debug_assert!(s.intra.len() >= n);
    debug_assert!(s.wdeg.len() >= n);
    s.intra[..n].fill(0.0);
    s.wdeg[..n].fill(0.0);

    let m_w = dg.total_weight;
    let inv_m = 1.0 / m_w;
    let m2 = 2.0 * m_w;
    let inv_m2_sq = 1.0 / (m2 * m2);

    let wdeg = &mut s.wdeg;
    let intra = &mut s.intra;
    let wd_src = &dg.weighted_deg;

    // SAFETY: partition IDs are non-negative throughout the PRISM pipeline
    // (initial partitions in [0,n), and all label-assignment paths copy
    // existing labels). Reinterpreting i32 → u32 yields the same bits; the
    // benefit is `movl` zero-extend instead of `movslq` sign-extend on every
    // access, saving an instruction per endpoint load in the edge loop.
    // u,v < n by DenseGraph construction; p.len() == n asserted above.
    let p_u = unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u32, p.len()) };

    unsafe {
        for i in 0..n {
            let c = *p_u.get_unchecked(i) as usize;
            *wdeg.get_unchecked_mut(c) += *wd_src.get_unchecked(i);
        }
        for &(u, v, w) in &dg.edges {
            let cu = *p_u.get_unchecked(u as usize);
            if cu == *p_u.get_unchecked(v as usize) {
                *intra.get_unchecked_mut(cu as usize) += w;
            }
        }
    }

    let mut q05 = 0.0;
    let mut q10 = 0.0;
    let mut q20 = 0.0;
    for c in 0..n {
        let vol = unsafe { *wdeg.get_unchecked(c) };
        if vol <= 0.0 {
            continue;
        }
        let ec = unsafe { *intra.get_unchecked(c) };
        let l = ec * inv_m;
        let v2 = vol * vol * inv_m2_sq;
        q05 += l - 0.5 * v2;
        q10 += l - v2;
        q20 += l - 2.0 * v2;
    }
    [-q05, -q10, -q20]
}

/// Label Propagation seed. Flat-array freq (size n), per-node
/// touched-list reset for O(deg) instead of O(n) cleanup.
pub fn lpa_partition(dg: &DenseGraph, iters: usize) -> DensePartition {
    use rand::RngExt;
    let mut rng = rand::rng();
    let n = dg.n;
    let mut p: DensePartition = (0..n).map(|i| i as CommunityId).collect();
    let mut order: Vec<usize> = (0..n).collect();

    let mut freq: Vec<u32> = vec![0; n];
    let mut touched: Vec<u32> = Vec::with_capacity(64);
    let mut tied: Vec<CommunityId> = Vec::with_capacity(8);

    let min_k = ((n as f64).sqrt().floor() as usize).max(4);
    for _ in 0..iters {
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
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
            let nbs = dg.neighbors(j);
            if nbs.is_empty() {
                continue;
            }
            touched.clear();
            let mut best = 0u32;
            for &nb in nbs {
                let c = p[nb as usize] as usize;
                if freq[c] == 0 {
                    touched.push(c as u32);
                }
                freq[c] += 1;
                if freq[c] > best {
                    best = freq[c];
                }
            }
            tied.clear();
            for &c in &touched {
                if freq[c as usize] == best {
                    tied.push(c as CommunityId);
                }
            }
            let new_c = if tied.len() == 1 {
                tied[0]
            } else {
                tied[rng.random_range(0..tied.len())]
            };
            for &c in &touched {
                freq[c as usize] = 0;
            }
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
