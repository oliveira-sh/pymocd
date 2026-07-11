//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::objectives::decomposed_modularity::calculate_objectives;
use crate::core::metaheuristics::helpers::objectives::metrics::Metrics;
use rand::rngs::ThreadRng;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

mod crossover;
mod generator;
mod mutation;

pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    mutation::mutate(partition, graph, mutation_rate);
}

pub fn ensemble_crossover(parents: &[&Partition], rng: &mut ThreadRng) -> Partition {
    crossover::ensemble_crossover(parents, rng)
}

pub fn get_fitness(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
    parallel: bool,
) -> Metrics {
    calculate_objectives(graph, partition, degrees, parallel)
}

pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    generator::generate_initial_population(graph, population_size)
}

pub fn get_modularity_from_partition(partition: &Partition, graph: &Graph) -> f64 {
    let metrics: Metrics =
        calculate_objectives(graph, partition, graph.precompute_degrees(), false);

    1.0 - metrics.inter - metrics.intra
}
