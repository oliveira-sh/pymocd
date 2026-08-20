//! Crowding distance and the truncation of the 2N pool back to `pop_size`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

use super::individual::Individual;
use super::sorting::fast_non_dominated_sort;

fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let n_obj = population[0].objectives.len();

    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for (idx, ind) in population.iter().enumerate() {
        rank_groups.entry(ind.rank).or_default().push(idx);
    }

    for indices in rank_groups.values() {
        if indices.len() <= 2 {
            for &i in indices {
                population[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }

        for obj_idx in 0..n_obj {
            let mut sorted = indices.clone();
            sorted.sort_unstable_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            population[sorted[0]].crowding_distance = f64::INFINITY;
            population[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;

            let obj_min = population[sorted[0]].objectives[obj_idx];
            let obj_max = population[sorted[sorted.len() - 1]].objectives[obj_idx];

            if (obj_max - obj_min).abs() > f64::EPSILON {
                let scale = 1.0 / (obj_max - obj_min);
                for i in 1..sorted.len() - 1 {
                    let prev_obj = population[sorted[i - 1]].objectives[obj_idx];
                    let next_obj = population[sorted[i + 1]].objectives[obj_idx];
                    population[sorted[i]].crowding_distance += (next_obj - prev_obj) * scale;
                }
            }
        }
    }
}

pub fn select_survivors(population: &mut Vec<Individual>, pop_size: usize) {
    fast_non_dominated_sort(population);
    calculate_crowding_distance(population);
    population.sort_unstable_by(|a, b| {
        a.rank.cmp(&b.rank).then_with(|| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });
    population.truncate(pop_size);
}
