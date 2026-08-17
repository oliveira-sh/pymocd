//! Graph builders shared by more than one of MOPSO's test modules.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

/// Two triangles joined by the single edge (2,3).
pub fn two_triangles() -> CsrGraph {
    let nodes: Vec<i32> = (0..6).collect();
    let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
    CsrGraph::from_edges(&nodes, &edges)
}

/// Two 5-cliques joined by the single edge (4,5).
pub fn two_cliques() -> CsrGraph {
    let mut e = Vec::new();
    for (lo, hi) in [(0, 5), (5, 10)] {
        for a in lo..hi {
            for b in (a + 1)..hi {
                e.push((a, b));
            }
        }
    }
    e.push((4, 5));
    CsrGraph::from_edges(&(0..10).collect::<Vec<i32>>(), &e)
}

/// `k` cliques of `s` vertices each, chained into a ring by one edge apiece.
/// The planted answer is `k` communities, and modularity's resolution limit
/// merges them once `k` grows — which is what CPM is here to avoid.
pub fn ring_of_cliques(k: i32, s: i32) -> CsrGraph {
    let nodes: Vec<i32> = (0..k * s).collect();
    let mut e = Vec::new();
    for c in 0..k {
        let lo = c * s;
        for a in lo..lo + s {
            for b in (a + 1)..lo + s {
                e.push((a, b));
            }
        }
        e.push((lo + s - 1, (lo + s) % (k * s)));
    }
    CsrGraph::from_edges(&nodes, &e)
}
