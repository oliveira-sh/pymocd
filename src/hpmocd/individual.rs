//! hpmocd/individual.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{Graph, Partition};
use crate::operators;
use rand::{prelude::*, rng};
use rayon::prelude::*;
use rustc_hash::FxHashSet as HashSet;

const ENSEMBLE_SIZE: usize = 4;

#[derive(Clone, Debug)]
pub struct Individual {
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
    pub fitness: f64,
}

impl Individual {
    pub fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
            fitness: f64::NEG_INFINITY,
        }
    }

    // Check if this individual dominates another
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

    #[inline(always)]
    pub fn calculate_fitness(&mut self) {
        self.fitness = 1.0 - self.objectives[0] - self.objectives[1];
    }
}

// Tournament selection with early return
#[inline]
pub fn tournament_selection(population: &[Individual], tournament_size: usize) -> &Individual {
    let mut rng: ThreadRng = rng();
    let best_idx: usize = rng.random_range(0..population.len());
    let mut best: &Individual = &population[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx: usize = rng.random_range(0..population.len());
        let candidate: &Individual = &population[candidate_idx];

        if candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance)
        {
            best = candidate;
        }
    }

    best
}

pub fn create_offspring(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
) -> Vec<Individual> {
    let pop_size = population.len();

    population
        .par_iter()
        .map(|_| {
            let mut rng = rng();
            let parents: Vec<&Individual> = {
                let mut unique_parents =
                    HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, Default::default());

                while unique_parents.len() < ENSEMBLE_SIZE {
                    let parent = tournament_selection(population, tournament_size);
                    unique_parents.insert(parent as *const Individual);
                }

                unique_parents
                    .into_iter()
                    .map(|ptr| unsafe { &*ptr })
                    .collect()
            };

            let parent_partitions: Vec<Partition> =
                parents.iter().map(|p| p.partition.clone()).collect();

            let mut child = if rng.random_bool(crossover_rate) {
                operators::ensemble_crossover(&parent_partitions, 1.0)
            } else {
                parent_partitions[0].clone()
            };

            operators::mutation(&mut child, graph, mutation_rate);
            Individual::new(child)
        })
        .take(pop_size)
        .collect()
}
