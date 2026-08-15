//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc2::{Genome, Labels};
use crate::core::graph::CsrGraph;

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


}
