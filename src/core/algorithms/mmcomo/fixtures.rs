//! Graph fixtures shared by more than one test module.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Graph;

/// Triangles `{0,1,2}` and `{3,4,5}` joined by the single bridge edge `(2,3)`.
pub fn two_triangle_edges() -> Vec<(i32, i32)> {
    vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
}

/// The same six nodes as an index-space [`Graph`].
pub fn two_triangles() -> Graph {
    let edges = [
        (0usize, 1usize),
        (1, 2),
        (0, 2),
        (3, 4),
        (4, 5),
        (3, 5),
        (2, 3),
    ];
    let n = 6;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in &edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    let m2: f64 = deg.iter().sum();
    Graph { n, adj, deg, m2 }
}
