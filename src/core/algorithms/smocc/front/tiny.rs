//! Tiny-community reabsorption: a community at or below the size threshold is
//! merged into its heaviest neighbour when that pull beats its own internal mass.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::graph::CsrGraph;

pub(super) fn refine_tiny(g: &CsrGraph, wadj: &[f64], part: &[i32], max_size: usize) -> Vec<i32> {
    let mut p = part.to_vec();
    for _ in 0..5 {
        let mut members: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
        for (u, &c) in p.iter().enumerate() {
            members.entry(c).or_default().push(u);
        }
        let tiny: Vec<i32> = members
            .iter()
            .filter(|(_, v)| v.len() <= max_size)
            .map(|(&c, _)| c)
            .collect();
        if tiny.is_empty() {
            break;
        }
        let mut moved = false;
        for c in tiny {
            let nodes = &members[&c];
            let mut internal = 0.0f64;
            let mut ext: FxHashMap<i32, f64> = FxHashMap::default();
            for &u in nodes {
                let start = g.xadj[u] as usize;
                let end = g.xadj[u + 1] as usize;
                for (&v, &w) in g.adj[start..end].iter().zip(&wadj[start..end]) {
                    let cv = p[v as usize];
                    if cv == c {
                        internal += w;
                    } else {
                        *ext.entry(cv).or_insert(0.0) += w;
                    }
                }
            }
            internal /= 2.0;
            let target = ext.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let sa = members.get(a.0).map_or(0, std::vec::Vec::len);
                        let sb = members.get(b.0).map_or(0, std::vec::Vec::len);
                        sa.cmp(&sb)
                    })
            });
            let Some((&tc, &te)) = target else { continue };
            if nodes.len() == 1 || internal == 0.0 || te > internal {
                for &u in nodes {
                    p[u] = tc;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::Labels;

    #[test]
    fn refine_tiny_merge_target_follows_edge_weights() {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 6),
            (3, 6),
        ];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let part: Labels = vec![0, 0, 0, 3, 3, 3, 6];
        let weight_edge = |w: &mut [f64], a: usize, b: usize, val: f64| {
            for (u, v) in [(a, b), (b, a)] {
                let start = g.xadj[u] as usize;
                let end = g.xadj[u + 1] as usize;
                for (p, &node) in g.adj.iter().enumerate().take(end).skip(start) {
                    if node as usize == v {
                        w[p] = val;
                    }
                }
            }
        };
        let mut w = vec![1.0; g.adj.len()];
        weight_edge(&mut w, 3, 6, 5.0);
        assert_eq!(refine_tiny(&g, &w, &part, 2)[6], 3, "heavier side lost");
        let mut w = vec![1.0; g.adj.len()];
        weight_edge(&mut w, 2, 6, 5.0);
        assert_eq!(refine_tiny(&g, &w, &part, 2)[6], 0, "heavier side lost");
    }
}
