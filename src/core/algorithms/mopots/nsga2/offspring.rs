//! Offspring production: binary tournament parents, ensemble crossover and mutation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::core::algorithms::mopots::operators::{ensemble_crossover, mutation};
use crate::core::algorithms::mopots::sampling::{bernoulli, slot_rng};
use crate::core::graph::{Graph, Partition};

use super::individual::Individual;

const ENSEMBLE_SIZE: usize = 4;

#[inline]
fn tournament_selection_index(
    population: &[Individual],
    tournament_size: usize,
    rng: &mut StdRng,
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

/// Builds `population.len()` children by ensemble crossover of tournament winners plus mutation.
/// `generation` salts the streams, so child `i` draws the same numbers on every run.
pub fn create_offspring(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    generation: usize,
) -> Vec<Individual> {
    let pop_size = population.len();
    if pop_size == 0 {
        return Vec::new();
    }

    // the clamp is what stops the distinct-parent loop below spinning on a tiny population
    let ensemble_size = ENSEMBLE_SIZE.min(pop_size);
    let crossover_dist = bernoulli(crossover_rate);
    let nodes = graph.nodes_vec();

    (0..pop_size)
        .into_par_iter()
        .map(|slot| {
            let mut rng = slot_rng(generation as u64, slot);

            let mut parent_indices: Vec<usize> = Vec::with_capacity(ensemble_size);
            while parent_indices.len() < ensemble_size {
                let parent_idx = tournament_selection_index(population, tournament_size, &mut rng);
                if !parent_indices.contains(&parent_idx) {
                    parent_indices.push(parent_idx);
                }
            }

            let parent_partitions: Vec<&Partition> = parent_indices
                .iter()
                .map(|&idx| &population[idx].partition)
                .collect();

            let mut child = if rng.sample(crossover_dist) {
                ensemble_crossover(&parent_partitions, nodes, &mut rng)
            } else {
                parent_partitions[rng.random_range(0..parent_partitions.len())].clone()
            };

            mutation(&mut child, graph, mutation_rate, &mut rng);
            Individual::new(child)
        })
        .collect()
}
