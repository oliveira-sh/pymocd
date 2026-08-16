//! The macro genome codec: seeded label propagation that decodes a centre set
//! into a partition, and the encoding that picks one centre per community.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use crate::core::algorithms::smocc::{Genome, Labels};

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

#[inline(never)]
fn propagate(g: &CsrGraph, wadj: &[f64], is_center: &[bool], lab: &mut [i32], n_slots: usize) {
    let n = g.n;
    let mut vote = vec![0.0f64; n_slots];
    let max_deg = g.deg.iter().copied().max().unwrap_or(0) as usize;
    let mut touched: Vec<u32> = vec![0u32; max_deg];
    debug_assert!(
        wadj.iter().all(|&w| w >= 0.0),
        "the fused argmax/reset below relies on non-negative edge weights"
    );

    const CLEAN: u8 = 0;
    const DIRTY: u8 = 1;
    const CENTER: u8 = 2;
    let mut state: Vec<u8> = is_center
        .iter()
        .map(|&c| if c { CENTER } else { DIRTY })
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
    for l in &mut lab {
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
    for (_, (node, _)) in best_node {
        genome[node] = 1;
    }
    genome
}

#[cfg(test)]
mod tests {
    use super::super::weights::init_weights;
    use super::*;
    use crate::core::algorithms::smocc::utils::fixtures::two_triangles;

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
        let mut uniq = lab;
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
        let mut uniq = lab;
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
        let mut uniq = lab;
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn decode_empty_genome_seeds_one_community() {
        let g = two_triangles();
        let w = init_weights(&g);
        let lab = decode(&g, &w, &vec![0u8; g.n]);
        let mut uniq = lab;
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
}
