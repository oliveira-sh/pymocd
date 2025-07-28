//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::leobenalg::individual::Individual;
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

pub fn fast_non_dominated_sort(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }
    fast_non_dominated_sort_nd(population);
}

fn fast_non_dominated_sort_nd(population: &mut [Individual]) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = population.len();
    let mut fronts: Vec<Vec<usize>> = Vec::with_capacity(n / 2);
    fronts.push(Vec::with_capacity(n / 2));

    let mut dominated_data = Vec::new();
    let mut dominated_ranges = Vec::with_capacity(n);
    let domination_count: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();

    let domination_relations: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::new();
            let mut count = 0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                if population[i].dominates(&population[j]) {
                    dominated.push(j);
                } else if population[j].dominates(&population[i]) {
                    count += 1;
                }
            }

            (dominated, count)
        })
        .collect();

    for (i, (dominated, count)) in domination_relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        dominated_ranges.push(start..dominated_data.len());
        domination_count[i].store(count, Ordering::Relaxed);

        if count == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }

    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = &fronts[front_idx];
        let next_front: Vec<usize> = current_front
            .par_iter()
            .fold(Vec::new, |mut acc, &i| {
                let range = &dominated_ranges[i];
                for &j in &dominated_data[range.start..range.end] {
                    let prev = domination_count[j].fetch_sub(1, Ordering::Relaxed);
                    if prev == 1 {
                        acc.push(j);
                    }
                }
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        front_idx += 1;
        if !next_front.is_empty() {
            for &j in &next_front {
                population[j].rank = front_idx + 1;
            }
            fronts.push(next_front);
        } else {
            break;
        }
    }
}

pub fn calculate_crowding_distance(population: &mut [Individual]) {
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

#[inline(always)]
pub fn q(ind: &Individual) -> f64 {
    1.0 - ind.objectives[0] - ind.objectives[1]
}
#[inline(always)]
pub fn max_q_selection(population: &[Individual]) -> &Individual {
    population
        .iter()
        .max_by(|a, b| q(a).partial_cmp(&q(b)).unwrap_or(Ordering::Equal))
        .expect("Empty population")
}
