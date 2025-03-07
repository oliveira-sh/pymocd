//! algorithms/pesa_ii/model_selection.rs
//! Implements the second phase of the algorithm (model selection)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::algorithms::pesa_ii::Solution;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

pub fn max_q_selection(archive: &[Solution]) -> &Solution {
    archive
        .iter()
        .max_by(|a, b| {
            let q_a = 1.0 - a.objectives[0] - a.objectives[1]; // 1 - intra - inter
            let q_b = 1.0 - b.objectives[0] - b.objectives[1];
            q_a.partial_cmp(&q_b).unwrap()
        })
        .unwrap()
}

/// Selects a solution from the real Pareto front based on the "max-min" distance criterion.
///
/// - `real_front`: A vector of non-dominated (Pareto) solutions from the real network.
/// - `random_fronts`: A vector of vectors containing non-dominated (Pareto) solutions from multiple random networks.
///
pub fn min_max_selection<'a>(
    real_front: &'a [Solution],
    random_fronts: &[Vec<Solution>],
) -> &'a Solution {
    // Keep track of the best solution and its distance
    let mut best_solution: Option<&Solution> = None;
    let mut best_max_min_distance = f64::MIN;

    // For each solution in the real front
    for real_sol in real_front {
        // For each random front, find the minimum distance to this solution
        let min_distances: Vec<f64> = random_fronts
            .iter()
            .map(|random_front| {
                random_front
                    .iter()
                    .map(|rand_sol| euclidean_distance(&real_sol.objectives, &rand_sol.objectives))
                    .fold(f64::MAX, |acc, val| acc.min(val))
            })
            .collect();

        // Get the minimum distance across all random fronts
        let max_min_distance = min_distances
            .iter()
            .fold(f64::MAX, |acc, &val| acc.min(val));

        // Update best solution if this one has a larger minimum distance
        if max_min_distance > best_max_min_distance {
            best_solution = Some(real_sol);
            best_max_min_distance = max_min_distance;
        }
    }

    best_solution.expect("Real Pareto front is empty.")
}
