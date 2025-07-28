//! hpmocd/individual.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::usize;

use crate::graph::{Graph, Partition};
use rand::distr::Bernoulli;
use rand::{prelude::*, rng};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashSet as HashSet};

use super::crossover;
use super::mutation;

const ENSEMBLE_SIZE: usize = 4;

pub type ObjVec = [f64; 2];

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
            objectives: [0.0, 0.0],
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
                crossover::ensemble_crossover(&parent_partitions, &mut rng)
            } else {
                parent_partitions[rng.random_range(0..parent_partitions.len())].clone()
            };

            mutation::mutate(&mut child, graph, mutation_rate);
            Individual::new(child)
        })
        .collect()
}
