//! MOCD-D model selection: the max-min distance decision rule.
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

/// The real front member that lies furthest from every random-network control
/// front (Shi 2012, Eqs. 3.9–3.11).
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
