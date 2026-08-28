//! Graph builders shared by more than one of CCM's test modules.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;

pub fn two_triangles() -> Graph {
    let mut g = Graph::new();
    for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
        g.add_edge(a, b);
    }
    g.finalize();
    g
}
