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
                    .fold(f64::MAX, |acc, val| acc.min(val))
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
use rustc_hash::FxHashSet;

/// Generates `num_networks` DEGREE-PRESERVING random networks (double-edge
/// swaps) as MOCD-D control fronts (Shi 2012, §3.2). Degree preservation is
/// required: the `inter` objective Σ(d_c/2m)² is degree-dependent, so an
/// Erdős–Rényi null shifts the whole control front on degree-heterogeneous
/// graphs and breaks the max-min selector.
pub fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    let norm = |a: NodeId, b: NodeId| if a <= b { (a, b) } else { (b, a) };
    (0..num_networks)
        .map(|_| {
            let mut edges: Vec<(NodeId, NodeId)> = original.edges.clone();
            let m = edges.len();
            let mut present: FxHashSet<(NodeId, NodeId)> =
                edges.iter().map(|&(a, b)| norm(a, b)).collect();
            let mut r = rng();
            // ~10 sweeps of attempted swaps mixes the topology while keeping
            // every node's degree exactly fixed.
            if m >= 2 {
                for _ in 0..(10 * m) {
                    let i = r.random_range(0..m);
                    let j = r.random_range(0..m);
                    if i == j {
                        continue;
                    }
                    let (a, b) = edges[i];
                    let (c, d) = edges[j];
                    // rewire (a-b),(c-d) -> (a-d),(c-b)
                    if a == d || c == b {
                        continue; // self-loop
                    }
                    let n1 = norm(a, d);
                    let n2 = norm(c, b);
                    if n1 == n2 || present.contains(&n1) || present.contains(&n2) {
                        continue; // multi-edge
                    }
                    present.remove(&norm(a, b));
                    present.remove(&norm(c, d));
                    present.insert(n1);
                    present.insert(n2);
                    edges[i] = (a, d);
                    edges[j] = (c, b);
                }
            }

            // Build through the real constructor so every derived field
            // (node_vec, degrees, adjacency_list, edge_lookup) is populated.
            let mut random_graph = Graph::new();
            for &(src, dst) in &edges {
                random_graph.add_edge(src, dst);
            }
            random_graph.finalize();
            random_graph
        })
        .collect()
}
