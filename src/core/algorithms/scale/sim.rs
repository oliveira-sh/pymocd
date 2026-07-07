//! sim — sparse, near-linear replacement for MMCoMO's dense n×n diffusion-kernel
//! similarity matrix.
//!
//! The dense baseline (`mmcomo`) materialises `SM = exp(beta·(A−D))` — an O(n²)
//! matrix built by an O(n³) Jacobi eigendecomposition — and uses it for three
//! things: decode (assign each node to its most-similar medoid centre), encode
//! (pick each community's medoid by intra-similarity), and the influence-step SM
//! update (Eq. 7). All three are reformulated here over the graph's edges so the
//! space is O(n + m) and the time per call is O(rounds·m) / O(m).
//!
//! The similarity is carried as a per-directed-adjacency-slot edge weight
//! `wadj` (length 2m, parallel to `CsrGraph::adj`), initialised to 1 and updated
//! by the influence step's co-membership consensus (`super::influence`). Because
//! `exp(beta·(A−D))` is, for the small beta the paper uses, a *local* kernel
//! dominated by short paths, replacing its argmax/medoid roles with weighted
//! graph propagation preserves the partitions it would have produced.

use crate::core::graph::CsrGraph;
use rayon::prelude::*;

use super::{Genome, Labels};

const UNSET: i32 = -1;

/// Max-degree node — the decode/encode fallback when a genome has no centre.
fn max_degree_node(g: &CsrGraph) -> usize {
    let mut best = 0usize;
    let mut best_deg = 0u32;
    for i in 0..g.n {
        if g.deg[i] > best_deg {
            best_deg = g.deg[i];
            best = i;
        }
    }
    best
}

/// Decode a medoid genome to a label vector — the sparse analogue of Eqs. 3-5.
///
/// Centres `CN = {i : genome[i] = 1}` keep their own id as label; every other
/// node is assigned to the centre it is most strongly connected to, via weighted
/// multi-source label propagation seeded at the centres. Each sweep a node adopts
/// the centre-label with the greatest summed incident edge weight among its
/// already-assigned neighbours (the dense kernel's argmax_{c} SM[i][c] is
/// dominated by exactly this short-path edge mass). Updates are applied in place
/// (asynchronous), so the flood advances many hops per sweep and converges in a
/// handful of sweeps without oscillating, while still — like the dense argmax
/// decode — assigning every node reachable from a centre.
/// Connected components that contain no centre are each collapsed into a single
/// community (their minimum node id), never per-node singletons.
pub fn decode(g: &CsrGraph, wadj: &[f64], genome: &Genome) -> Labels {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    let mut is_center = vec![false; n];
    let mut any = false;
    for i in 0..n {
        if genome.get(i).copied().unwrap_or(0) != 0 {
            is_center[i] = true;
            any = true;
        }
    }
    if !any {
        // Eq. 3 fallback (s = 0): seed the max-degree node as the sole centre.
        is_center[max_degree_node(g)] = true;
    }

    let mut lab: Vec<i32> = (0..n)
        .map(|i| if is_center[i] { i as i32 } else { UNSET })
        .collect();

    // Dense vote table indexed by centre-id (labels are node ids < n), reset via
    // a touched list so each sweep is O(degree) per node — no hashing.
    let mut vote = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::with_capacity(64);

    // ASYNCHRONOUS label propagation: each node's new label is written in place,
    // so later nodes in a sweep already see it — the centre flood advances many
    // hops per sweep and (unlike synchronous LP) cannot two-cycle oscillate, so
    // it converges in a handful of sweeps even on dense graphs. The `changed`
    // break exits the moment the partition is stable; the `n` bound is only a
    // runaway guard (a sweep visits the whole node order regardless of diameter).
    for _ in 0..n {
        let mut changed = false;
        for u in 0..n {
            if is_center[u] {
                continue; // centres are fixed seeds
            }
            touched.clear();
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            let mut best = lab[u];
            let mut best_w = if lab[u] != UNSET { 0.0 } else { -1.0 };
            // Tally weighted votes over assigned neighbours (already-updated this
            // sweep for v < u — that is the asynchronous flood).
            for p in start..end {
                let v = g.adj[p] as usize;
                let lv = lab[v];
                if lv == UNSET {
                    continue;
                }
                let li = lv as usize;
                if vote[li] == 0.0 {
                    touched.push(li);
                }
                vote[li] += wadj[p];
            }
            // Keep current label on ties (stability); otherwise strict argmax.
            if best != UNSET {
                best_w = vote[best as usize];
            }
            for &c in &touched {
                if vote[c] > best_w {
                    best_w = vote[c];
                    best = c as i32;
                }
            }
            for &c in &touched {
                vote[c] = 0.0;
            }
            if best != lab[u] {
                lab[u] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Any node still UNSET lies in a connected component that holds no centre
    // (such a node's neighbours are all themselves UNSET, else the flood would
    // have reached it). Collapse each such component into ONE community — its
    // minimum node id — via an in-place min-id flood, restricted to the leftover
    // nodes so reached nodes keep their centre label.
    if lab.iter().any(|&l| l == UNSET) {
        let leftover: Vec<bool> = lab.iter().map(|&l| l == UNSET).collect();
        for u in 0..n {
            if leftover[u] {
                lab[u] = u as i32;
            }
        }
        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                if !leftover[u] {
                    continue;
                }
                let mut m = lab[u];
                for &v in g.neighbors(u) {
                    if lab[v as usize] < m {
                        m = lab[v as usize];
                    }
                }
                if m != lab[u] {
                    lab[u] = m;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    lab
}

/// Encode a label vector to a medoid genome — the sparse analogue of Eq. 8.
///
/// The dense rule picks, per community, the member of maximal summed intra-block
/// similarity. Its sparse counterpart is the member of maximal **weighted
/// internal degree** (summed incident edge weight to same-community neighbours):
/// the most internally-central node, computed in one O(m) pass. Singletons mark
/// themselves as the centre.
pub fn encode(g: &CsrGraph, wadj: &[f64], labels: &Labels) -> Genome {
    let n = g.n;
    let mut genome: Genome = vec![0u8; n];
    if n == 0 {
        return genome;
    }

    // Weighted internal degree per node.
    let mut internal = vec![0.0f64; n];
    let mut size: rustc_hash::FxHashMap<i32, usize> = rustc_hash::FxHashMap::default();
    for u in 0..n {
        *size.entry(labels[u]).or_insert(0) += 1;
        let start = g.xadj[u] as usize;
        let end = g.xadj[u + 1] as usize;
        let cu = labels[u];
        let mut acc = 0.0;
        for p in start..end {
            let v = g.adj[p] as usize;
            if labels[v] == cu {
                acc += wadj[p];
            }
        }
        internal[u] = acc;
    }

    // Per community, the node of maximal internal weight is the medoid.
    let mut best_node: rustc_hash::FxHashMap<i32, (usize, f64)> = rustc_hash::FxHashMap::default();
    for u in 0..n {
        let c = labels[u];
        match best_node.get(&c) {
            Some(&(_, w)) if w >= internal[u] => {}
            _ => {
                best_node.insert(c, (u, internal[u]));
            }
        }
    }
    for (c, (node, _)) in best_node {
        // Singletons (size 1) trivially mark themselves; otherwise the medoid.
        let _ = c;
        genome[node] = 1;
    }
    genome
}

/// Initial edge weights: 1 for every directed adjacency slot (length 2m).
pub fn init_weights(g: &CsrGraph) -> Vec<f64> {
    vec![1.0f64; g.adj.len()]
}

/// Influence-step weight update (Eq. 7 analogue, sparse).
///
/// Accumulates, per directed adjacency slot, the fraction of `elites` whose two
/// endpoints share a community (the edge-restricted micro-elite voting matrix
/// `SM^v`), then blends it into `wadj` with `wadj* = (1−rho)·wadj + rho·cov`.
/// O(|elites|·m) time, O(m) extra space — the dense O(|PF|·n²) triple loop made
/// near-linear. `cov` is symmetric across an edge's two slots, so `wadj` stays
/// symmetric and `decode`/`encode` see a consistent similarity.
pub fn update_weights(g: &CsrGraph, wadj: &mut [f64], elites: &[&Labels], rho: f64) {
    let two_m = g.adj.len();
    if two_m == 0 || elites.is_empty() {
        return;
    }
    let pf = elites.len() as f64;
    // Build `cov` in parallel over source nodes. Slot p belongs to exactly one
    // source node u (p ∈ [xadj[u], xadj[u+1])), and `flat_map_iter` emits slots in
    // that same CSR order, so the parallel build is race-free and order-exact.
    let cov: Vec<f64> = (0..g.n)
        .into_par_iter()
        .flat_map_iter(|u| {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            (start..end).map(move |p| {
                let v = g.adj[p] as usize;
                let mut c = 0.0;
                for e in elites {
                    if e[u] == e[v] {
                        c += 1.0 / pf;
                    }
                }
                c
            })
        })
        .collect();
    wadj.par_iter_mut().zip(cov.par_iter()).for_each(|(w, &c)| {
        *w = (1.0 - rho) * *w + rho * c;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::CsrGraph;

    // Two triangles {0,1,2},{3,4,5} joined by bridge (2,3).
    fn two_triangles() -> CsrGraph {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        CsrGraph::from_edges(&nodes, &edges)
    }

    #[test]
    fn decode_one_centre_per_triangle_splits_two() {
        let g = two_triangles();
        let w = init_weights(&g);
        let mut genome = vec![0u8; g.n];
        genome[0] = 1;
        genome[3] = 1;
        let lab = decode(&g, &w, &genome);
        // Bridge node 2 follows triangle 1, node 3-region the other.
        assert_eq!(lab[0], lab[1]);
        assert_eq!(lab[1], lab[2]);
        assert_eq!(lab[3], lab[4]);
        assert_eq!(lab[4], lab[5]);
        assert_ne!(lab[0], lab[3]);
        let mut uniq = lab.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn decode_high_diameter_single_centre_no_singletons() {
        // 200-node path with one centre at node 0: every node must join the
        // centre's community (1 community), not fragment into distance-singletons
        // (regression for the removed 64-round cap).
        let n = 200i32;
        let nodes: Vec<i32> = (0..n).collect();
        let edges: Vec<(i32, i32)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = init_weights(&g);
        let mut genome = vec![0u8; g.n];
        // CsrGraph interns nodes in `nodes` order, so dense id 0 == file id 0.
        genome[0] = 1;
        let lab = decode(&g, &w, &genome);
        let mut uniq = lab.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 1, "high-diameter path fragmented into {} communities", uniq.len());
    }

    #[test]
    fn decode_centreless_component_is_one_community() {
        // Two disjoint triangles; genome centres only the first. The second
        // (centreless) component must collapse to ONE community, not 3 singletons.
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = init_weights(&g);
        let mut genome = vec![0u8; g.n];
        genome[0] = 1; // centre only in the first triangle
        let lab = decode(&g, &w, &genome);
        assert_eq!(lab[3], lab[4]);
        assert_eq!(lab[4], lab[5]);
        assert_ne!(lab[0], lab[3]);
        let mut uniq = lab.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn decode_empty_genome_seeds_one_community() {
        let g = two_triangles();
        let w = init_weights(&g);
        let lab = decode(&g, &w, &vec![0u8; g.n]);
        let mut uniq = lab.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 1);
    }

    #[test]
    fn encode_picks_internal_node_per_community() {
        let g = two_triangles();
        let w = init_weights(&g);
        let labels = vec![0, 0, 0, 9, 9, 9];
        let genome = encode(&g, &w, &labels);
        let centers: Vec<usize> = (0..g.n).filter(|&i| genome[i] == 1).collect();
        assert_eq!(centers.len(), 2);
        assert_eq!(centers.iter().filter(|&&c| c < 3).count(), 1);
        assert_eq!(centers.iter().filter(|&&c| c >= 3).count(), 1);
    }

    #[test]
    fn encode_decode_roundtrip_preserves_blocks() {
        let g = two_triangles();
        let w = init_weights(&g);
        let labels = vec![0, 0, 0, 9, 9, 9];
        let genome = encode(&g, &w, &labels);
        let back = decode(&g, &w, &genome);
        assert_eq!(back[0], back[1]);
        assert_eq!(back[1], back[2]);
        assert_eq!(back[3], back[4]);
        assert_eq!(back[4], back[5]);
        assert_ne!(back[0], back[3]);
    }

    #[test]
    fn update_weights_raises_intra_lowers_inter() {
        let g = two_triangles();
        let mut w = init_weights(&g);
        let elite: Labels = vec![0, 0, 0, 1, 1, 1];
        update_weights(&g, &mut w, &[&elite], 1.0);
        // With rho=1 and one elite, intra-edge slots → 1, bridge slots → 0.
        for u in 0..g.n {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            for p in start..end {
                let v = g.adj[p] as usize;
                let same = (u < 3) == (v < 3);
                assert!((w[p] - if same { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }
}
