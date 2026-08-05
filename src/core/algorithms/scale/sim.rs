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

pub fn decode(g: &CsrGraph, wadj: &[f64], genome: &Genome) -> Labels {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    let mut is_center = vec![false; n];
    let mut any = false;
    for (flag, &gene) in is_center.iter_mut().zip(genome) {
        if gene != 0 {
            *flag = true;
            any = true;
        }
    }
    if !any {
        is_center[max_degree_node(g)] = true;
    }

    let mut lab: Vec<i32> = (0..n)
        .map(|i| if is_center[i] { i as i32 } else { UNSET })
        .collect();

    let mut vote = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::with_capacity(64);

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
    let mut dirty: Vec<bool> = is_center.iter().map(|&c| !c).collect();

    for _ in 0..n {
        let mut changed = false;
        for u in 0..n {
            if !dirty[u] {
                continue;
            }
            dirty[u] = false;
            touched.clear();
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            let mut best = lab[u];
            let mut best_w = if lab[u] != UNSET { 0.0 } else { -1.0 };
            for (&v, &w) in g.adj[start..end].iter().zip(&wadj[start..end]) {
                let lv = lab[v as usize];
                if lv == UNSET {
                    continue;
                }
                let li = lv as usize;
                if vote[li] == 0.0 {
                    touched.push(li);
                }
                vote[li] += w;
            }
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
                for &v in &g.adj[start..end] {
                    let v = v as usize;
                    if !is_center[v] {
                        dirty[v] = true;
                    }
                }
            }
        }
        if !changed {
            break;
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
