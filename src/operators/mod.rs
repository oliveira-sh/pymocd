//! operators/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{Graph, Partition};
use metrics::Metrics;
use rand::rngs::ThreadRng;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

pub mod metrics;

mod crossover;
mod mutation;
mod objective;
mod population;

/// Represents the convergence criteria and state for the genetic algorithm
#[derive(Debug)]
pub struct ConvergenceCriteria {
    current_best_fitness: f64,       // Current best fitness value found
    generations_unchanged: usize,    // Number of generations without improvement
    max_stagnant_generations: usize, // Maximum allowed generations without improvement
    tolerance: f64,                  // Numerical tolerance for fitness comparison
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        ConvergenceCriteria {
            current_best_fitness: f64::MIN,
            generations_unchanged: 0,
            max_stagnant_generations: 100,
            tolerance: 1e-6,
        }
    }
}

impl ConvergenceCriteria {
    /// Checks if the algorithm has converged based on the latest fitness value
    /// Returns true if convergence criteria are met, false otherwise
    pub fn has_converged(&mut self, new_fitness: f64) -> bool {
        // Check if there's a significant improvement
        let has_improved = (new_fitness - self.current_best_fitness).abs() > self.tolerance;

        if has_improved {
            // Reset counter if we found a better solution
            self.current_best_fitness = new_fitness;
            self.generations_unchanged = 0;
            return false;
        }

        self.generations_unchanged += 1;
        if self.generations_unchanged >= self.max_stagnant_generations {
            return true;
        }

        false
    }

    pub fn get_best_fitness(&self) -> f64 {
        self.current_best_fitness
    }
}

pub fn crossover(parent1: &Partition, parent2: &Partition, crossover_rate: f64) -> Partition {
    crossover::two_point_crossover(parent1, parent2, crossover_rate)
}

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
) -> metrics::Metrics {
    objective::calculate_objectives(graph, partition, degrees, parallel)
}

pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    population::generate_initial_population(graph, population_size)
}

pub fn get_modularity_from_partition(partition: &Partition, graph: &Graph) -> f64 {
    let metrics: Metrics =
        objective::calculate_objectives(graph, partition, &graph.precompute_degrees(), false);

    metrics.get_modularity()
}
