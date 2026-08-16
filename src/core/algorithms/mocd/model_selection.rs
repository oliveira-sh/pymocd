//! Model selection phase for Shi-MOCD (MOCD-D, Shi 2012 §3.2).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::pesa2::Solution;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Selects a solution from the real Pareto front by the "max-min" distance
/// criterion against the random-network control fronts.
pub fn min_max_selection<'a>(
    real_front: &'a [Solution],
    random_fronts: &[Vec<Solution>],
) -> &'a Solution {
    let mut best_solution: Option<&Solution> = None;
    let mut best_max_min_distance = f64::MIN;

    for real_sol in real_front {
        let min_distances: Vec<f64> = random_fronts
            .iter()
            .map(|random_front| {
                random_front
                    .iter()
                    .map(|rand_sol| euclidean_distance(&real_sol.objectives, &rand_sol.objectives))
                    .fold(f64::MAX, f64::min)
            })
            .collect();

        let max_min_distance = min_distances
            .iter()
            .fold(f64::MAX, |acc, &val| acc.min(val));

        if max_min_distance > best_max_min_distance {
            best_solution = Some(real_sol);
            best_max_min_distance = max_min_distance;
        }
    }

    best_solution.expect("Real Pareto front is empty.")
}

use crate::core::graph::{Graph, NodeId};
use rand::{RngExt, rng};
use std::collections::HashSet;

/// Generates `num_networks` Erdős–Rényi `G(n, m)` random networks — same node
/// and edge counts as `original`, uniformly random simple edges — as MOCD-D
/// control fronts ("random networks with the same scale", Shi 2012 §3.2).
/// A degree-preserving (double-edge-swap) null was considered but is not what
/// Shi describes.
pub fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    let nodes: Vec<NodeId> = original.nodes_vec().clone();
    let n = nodes.len();
    let m = original.edges.len();
    (0..num_networks)
        .map(|_| {
            let mut r = rng();
            let mut present: HashSet<(NodeId, NodeId)> = HashSet::with_capacity(m);
            // Build through the real constructor so every derived field
            // (node_vec, degrees, adjacency_list, edge_lookup) is populated;
            // pre-insert all nodes so the node count matches `original`.
            let mut random_graph = Graph::new();
            for &node in &nodes {
                random_graph.nodes.insert(node);
                random_graph.adjacency_list.entry(node).or_default();
            }
            while present.len() < m {
                let a = nodes[r.random_range(0..n)];
                let b = nodes[r.random_range(0..n)];
                if a == b {
                    continue; // self-loop
                }
                let key = if a <= b { (a, b) } else { (b, a) };
                if present.insert(key) {
                    random_graph.add_edge(key.0, key.1);
                }
            }
            random_graph.finalize();
            random_graph
        })
        .collect()
}
