//! Graph builders shared by more than one of SMOCC's test modules.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

pub fn two_clique_edges() -> Vec<(i32, i32)> {
    let mut e = Vec::new();
    for (lo, hi) in [(0, 5), (5, 10)] {
        for a in lo..hi {
            for b in (a + 1)..hi {
                e.push((a, b));
            }
        }
    }
    e.push((4, 5));
    e
}

pub fn two_triangles() -> CsrGraph {
    let nodes: Vec<i32> = (0..6).collect();
    let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
    CsrGraph::from_edges(&nodes, &edges)
}
