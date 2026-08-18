//! Test fixture: two triangles 0-1-2 and 3-4-5 joined by the single bridge (2,3).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;

pub(super) fn two_triangles() -> Graph {
    let mut g = Graph::new();
    for &(u, v) in &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)] {
        g.add_edge(u, v);
    }
    g.finalize();
    g
}
