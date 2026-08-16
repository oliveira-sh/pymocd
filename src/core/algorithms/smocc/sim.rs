use crate::core::graph::CsrGraph;
use rayon::prelude::*;

use super::{Genome, Labels};

const UNSET: i32 = -1;

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

/// Asynchronous weighted multi-source label propagation over centre SLOTS.
/// `lab` enters and leaves in slot space; `n_slots` is the number of centres.
/// Kept out of line so the slot bookkeeping in `decode` cannot steal the
/// register holding `vote`'s base pointer - inlining it costs more than the
/// smaller vote table saves.
#[inline(never)]
fn propagate(g: &CsrGraph, wadj: &[f64], is_center: &[bool], lab: &mut [i32], n_slots: usize) {
    let n = g.n;
    let mut vote = vec![0.0f64; n_slots];
    // Pre-sized so the "push" is a store plus an increment. The `Vec::push` it
    // replaces carried a reallocating call in the hot edge loop, and the
    // possible call made the register allocator spill the incident weight and
    // the vote cell to the stack and reload them on every single edge.
    // The bound is the maximum degree, NOT `n_slots`: the `vote[li] == 0.0`
    // guard below re-records a slot whose running sum is still exactly 0.0, so
    // an incident weight of 0.0 makes entries repeat and one node can record up
    // to `deg(u)` of them.
    let max_deg = g.deg.iter().copied().max().unwrap_or(0) as usize;
    let mut touched: Vec<u32> = vec![0u32; max_deg];
    debug_assert!(
        wadj.iter().all(|&w| w >= 0.0),
        "the fused argmax/reset below relies on non-negative edge weights"
    );

    // Active set. A node's next label is a pure function of its own label and
    // of the labels/weights on its incident edges (`wadj` is fixed for the
    // whole decode), and that function is idempotent: it re-selects the label
    // it just wrote. A node none of whose inputs moved since its last
    // evaluation therefore recomputes the same label and cannot set `changed`,
    // so skipping it is observationally identical to rescanning it. The visit
    // order stays ascending `0..n`, keeping the Gauss-Seidel trajectory
    // intact - a neighbour `v > u` dirtied while evaluating `u` is still
    // reached later in the same sweep. Centres are fixed seeds, so they are
    // permanently clean. Requires a symmetric adjacency (CsrGraph::from_edges
    // pushes both directions), so every reader of a label that moved is
    // reachable from the node that moved it.
    //
    // The active flag and the centre flag live in ONE byte array: CLEAN(0) and
    // DIRTY(1) for ordinary nodes, CENTER(2) for the permanently clean seeds.
    // Marking a neighbour is then `(s >> 1) + 1`, which maps 0 -> 1, 1 -> 1 and
    // 2 -> 2: it dirties ordinary nodes, leaves already-dirty ones alone and
    // cannot wake a centre. That is exactly the old
    // `if !is_center[v] { dirty[v] = true }` with one array instead of two -
    // half the resident bytes, one scattered load per marked edge instead of
    // two, and no branch to mispredict.
    const CLEAN: u8 = 0;
    const DIRTY: u8 = 1;
    let mut state: Vec<u8> = is_center
        .iter()
        .map(|&c| if c { 2 } else { DIRTY })
        .collect();

    for _ in 0..n {
        let mut changed = false;
        for u in 0..n {
            if state[u] != DIRTY {
                continue;
            }
            state[u] = CLEAN;
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            let cur = lab[u];
            let mut nt = 0usize;
            for (&v, &w) in g.adj[start..end].iter().zip(&wadj[start..end]) {
                // `UNSET` is -1, whose u32 image is u32::MAX, and slot ids are
                // dense in `0..n_slots` with `n_slots <= n < 2^32` (node ids are
                // stored as u32 in `adj`). One unsigned range test therefore
                // does the UNSET filter and the vote bound check at once, and
                // leaves the indexing below provably in range.
                let li = lab[v as usize] as u32 as usize;
                if li >= vote.len() {
                    continue;
                }
                if vote[li] == 0.0 {
                    touched[nt] = li as u32;
                    nt += 1;
                }
                vote[li] += w;
            }
            let mut best = cur;
            let mut best_w = if cur != UNSET {
                vote[cur as usize]
            } else {
                -1.0
            };
            // Argmax and reset are fused: same first-touch order, same strict
            // `>` first-to-the-max tie-break. A repeated slot reads 0.0 on its
            // second visit instead of the sum, which is equivalent exactly
            // because weights are non-negative (asserted above): the sum then
            // already beat `best_w` on the first visit, so the 0.0 cannot win
            // and cannot displace it.
            for &c in &touched[..nt] {
                let ci = c as usize;
                let vw = vote[ci];
                vote[ci] = 0.0;
                if vw > best_w {
                    best_w = vw;
                    best = c as i32;
                }
            }
            if best != cur {
                lab[u] = best;
                changed = true;
                for &v in &g.adj[start..end] {
                    let v = v as usize;
                    state[v] = (state[v] >> 1) + 1;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub fn decode(g: &CsrGraph, wadj: &[f64], genome: &Genome) -> Labels {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    let mut is_center = vec![false; n];
    let mut n_centers = 0usize;
    for (flag, &gene) in is_center.iter_mut().zip(genome) {
        if gene != 0 {
            *flag = true;
            n_centers += 1;
        }
    }
    if n_centers == 0 {
        is_center[max_degree_node(g)] = true;
        n_centers = 1;
    }

    // Centres are numbered 0..c in ascending node-id order and `lab` carries
    // those slot ids for the duration of the propagation, so the vote table is
    // only c wide (c ~ sqrt(n) under the macro centre cap) instead of n, and
    // stays cache- and TLB-resident. Slots are an order-preserving bijection
    // with the centre node ids, so every vote sum, every first-touch position
    // and every comparison is unchanged. They are mapped back to node ids
    // before the leftover pass, which reads labels as node ids.
    let mut center_node: Vec<i32> = Vec::with_capacity(n_centers);
    let mut lab: Labels = vec![UNSET; n];
    for i in 0..n {
        if is_center[i] {
            lab[i] = center_node.len() as i32;
            center_node.push(i as i32);
        }
    }

    propagate(g, wadj, &is_center, &mut lab, n_centers);

    debug_assert!(
        lab.iter()
            .all(|&l| l == UNSET || (l as usize) < center_node.len()),
        "propagate left a non-slot label behind"
    );
    for l in lab.iter_mut() {
        if *l != UNSET {
            *l = center_node[*l as usize];
        }
    }

    if lab.contains(&UNSET) {
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

pub fn encode(g: &CsrGraph, wadj: &[f64], labels: &Labels) -> Genome {
    let n = g.n;
    let mut genome: Genome = vec![0u8; n];
    if n == 0 {
        return genome;
    }

    let mut internal = vec![0.0f64; n];
    let mut size: rustc_hash::FxHashMap<i32, usize> = rustc_hash::FxHashMap::default();
    for u in 0..n {
        *size.entry(labels[u]).or_insert(0) += 1;
        let start = g.xadj[u] as usize;
        let end = g.xadj[u + 1] as usize;
        let cu = labels[u];
        let mut acc = 0.0;
        for (&v, &w) in g.adj[start..end].iter().zip(&wadj[start..end]) {
            if labels[v as usize] == cu {
                acc += w;
            }
        }
        internal[u] = acc;
    }

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
        let _ = c;
        genome[node] = 1;
    }
    genome
}

pub fn init_weights(g: &CsrGraph) -> Vec<f64> {
    vec![1.0f64; g.adj.len()]
}

pub fn update_weights(g: &CsrGraph, wadj: &mut [f64], elites: &[&Labels], rho: f64) {
    let two_m = g.adj.len();
    if two_m == 0 || elites.is_empty() {
        return;
    }
    let pf = elites.len() as f64;
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
        let n = 200i32;
        let nodes: Vec<i32> = (0..n).collect();
        let edges: Vec<(i32, i32)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = init_weights(&g);
        let mut genome = vec![0u8; g.n];
        genome[0] = 1;
        let lab = decode(&g, &w, &genome);
        let mut uniq = lab.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            1,
            "high-diameter path fragmented into {} communities",
            uniq.len()
        );
    }

    #[test]
    fn decode_centreless_component_is_one_community() {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = init_weights(&g);
        let mut genome = vec![0u8; g.n];
        genome[0] = 1;
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
        for u in 0..g.n {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            for (&v, &wp) in g.adj[start..end].iter().zip(&w[start..end]) {
                let same = (u < 3) == ((v as usize) < 3);
                assert!((wp - if same { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }
}
