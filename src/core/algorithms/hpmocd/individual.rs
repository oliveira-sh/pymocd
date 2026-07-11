//! Shared NSGA core: the engine-agnostic `Individual` (label-map `Partition` +
//! `Vec<f64>` objectives), Pareto dominance, fast non-dominated sorting, and
//! tournament-based offspring generation (Deb et al. 2002). Reused by **both**
//! `nsga2` and `nsga3` — each engine adds only its own survivor-selection step
//! (crowding distance vs. reference-point niching). Kept here, not inside either
//! engine, so neither engine has to reach into the other for these primitives.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, Partition};
use super::operators;
use rand::distr::Bernoulli;
use rand::{prelude::*, rng};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashSet as HashSet};

const ENSEMBLE_SIZE: usize = 4;
pub const TOURNAMENT_SIZE: usize = 2;

pub type ObjVec = Vec<f64>;

#[derive(Clone, Debug)]
pub struct Individual {
    pub partition: Partition,
    pub objectives: ObjVec,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Individual {
    pub fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }
    #[inline(always)]
    pub fn dominates(&self, other: &Individual) -> bool {
        let mut at_least_one_better = false;

        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }
}

#[inline]
fn tournament_selection_index(
    population: &[Individual],
    tournament_size: usize,
    rng: &mut ThreadRng,
) -> usize {
    let mut best_idx = rng.random_range(0..population.len());
    let mut best = &population[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx = rng.random_range(0..population.len());
        let candidate = &population[candidate_idx];

        if candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance)
        {
            best = candidate;
            best_idx = candidate_idx;
        }
    }

    best_idx
}

pub fn create_offspring(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
) -> Vec<Individual> {
    let pop_size = population.len();
    let crossover_dist = Bernoulli::new(crossover_rate).unwrap();
    let parent_indices: Vec<Vec<usize>> = (0..pop_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rng();
            let mut unique_parents =
                HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, FxBuildHasher);

            while unique_parents.len() < ENSEMBLE_SIZE {
                let parent_idx = tournament_selection_index(population, tournament_size, &mut rng);
                unique_parents.insert(parent_idx);
            }

            unique_parents.into_iter().collect()
        })
        .collect();

    parent_indices
        .into_par_iter()
        .map(|parent_idx_vec| {
            let mut rng = rng();
            let parent_partitions: Vec<&Partition> = parent_idx_vec
                .iter()
                .map(|&idx| &population[idx].partition)
                .collect();

            let mut child = if crossover_dist.sample(&mut rng) {
                operators::ensemble_crossover(&parent_partitions, &mut rng)
            } else {
                parent_partitions[rng.random_range(0..parent_partitions.len())].clone()
            };

            operators::mutation(&mut child, graph, mutation_rate);
            Individual::new(child)
        })
        .collect()
}

pub fn fast_non_dominated_sort(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }
    fast_non_dominated_sort_nd(population);
}

fn fast_non_dominated_sort_nd(population: &mut [Individual]) {
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
