//! operators/selection.rs
//! Make population selections in the Genetic Algorithm
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::Partition;
use crate::operators::metrics::Metrics;
use rand::seq::SliceRandom;
use rayon::prelude::*;

#[allow(dead_code)]
pub fn selection(
    population: Vec<Partition>,
    fitnesses: Vec<Metrics>,
    pop_size: usize,
) -> Vec<Partition> {
    let mut population_with_fitness: Vec<_> = population.into_par_iter().zip(fitnesses).collect();

    // Sort by modularity (descending order)
    population_with_fitness
        .sort_by(|(_, a), (_, b)| b.modularity.partial_cmp(&a.modularity).unwrap());

    population_with_fitness
        .into_par_iter()
        .take(pop_size / 2)
        .map(|(p, _)| p)
        .collect()
}

// Tournament selection
#[allow(dead_code)]
pub fn optimized_selection(
    population: Vec<Partition>,
    fitnesses: Vec<Metrics>,
    pop_size: usize,
    tournament_size: usize,
) -> Vec<Partition> {
    let mut rng = rand::thread_rng();
    let population_with_fitness: Vec<_> = population.into_par_iter().zip(fitnesses).collect();

    let mut selected = Vec::with_capacity(pop_size / 2);

    while selected.len() < pop_size / 2 {
        // tournament_size random individuals
        let tournament: Vec<_> = (0..tournament_size)
            .map(|_| population_with_fitness.choose(&mut rng).unwrap())
            .collect();

        // Find the winner, (highest modularity)
        let winner = tournament
            .iter()
            .max_by(|(_, a), (_, b)| a.modularity.partial_cmp(&b.modularity).unwrap())
            .unwrap();

        selected.push(winner.0.clone());
    }

    selected
}
