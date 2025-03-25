//! Implements the Pareto Envelope-based Selection Algorithm II (PESA-II)
//! This Source Code Form is subject to the terms of the GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos.
//! If a copy of the GPL was not distributed with this file, you can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod individual;
mod utils;

use crate::graph::{self, Graph, Partition};
use crate::operators;
use crate::utils::{build_graph, get_edges, normalize_community_ids};
use individual::{Individual, create_offspring};
use utils::{calculate_crowding_distance, fast_non_dominated_sort, max_q_selection};

use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::cmp::Ordering;
use std::collections::HashMap;

const TOURNAMENT_SIZE: usize = 2;

#[pyclass]
pub struct HpMocd {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
}

/* Private (Not exposed to py user) */
impl HpMocd {
    fn evaluate_population(
        &self,
        individuals: &mut [Individual],
        graph: &Graph,
        degrees: &HashMap<i32, usize, FxBuildHasher>,
    ) {
        individuals.par_iter_mut().for_each(|ind| {
            let metrics = operators::get_fitness(graph, &ind.partition, degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
            ind.calculate_fitness();
        });
    }

    fn update_population_sort_and_truncate(
        &self,
        individuals: &mut Vec<Individual>,
        pop_size: usize,
    ) {
        fast_non_dominated_sort(individuals);
        calculate_crowding_distance(individuals);
        individuals.sort_unstable_by(|a, b| {
            a.rank.cmp(&b.rank).then_with(|| {
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap_or(Ordering::Equal)
            })
        });
        individuals.truncate(pop_size);
    }

    fn envolve(&self) -> Vec<Individual> {
        if self.debug_level >= 1 {
            self.graph.print();
        }

        let degrees = &self.graph.precompute_degrees();
        let mut individuals: Vec<Individual> =
            operators::generate_population(&self.graph, self.pop_size)
                .into_par_iter()
                .map(Individual::new)
                .collect();
        self.evaluate_population(&mut individuals, &self.graph, degrees);

        let mut max_local = operators::ConvergenceCriteria::default();
        for generation in 0..self.num_gens {
            let len = individuals.len();
            self.update_population_sort_and_truncate(&mut individuals, len);

            // Create offspring and evaluate them.
            let mut offspring = create_offspring(
                &individuals,
                &self.graph,
                self.cross_rate,
                self.mut_rate,
                TOURNAMENT_SIZE,
            );
            self.evaluate_population(&mut offspring, &self.graph, degrees);

            // Combine and prepare for environmental selection.
            individuals.extend(offspring);
            self.update_population_sort_and_truncate(&mut individuals, self.pop_size);

            // Record best fitness.
            let best_fitness = individuals
                .par_iter()
                .map(|ind| ind.fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f64::NEG_INFINITY);

            if max_local.has_converged(best_fitness) {
                if self.debug_level >= 1 {
                    println!("[evolutionary_phase]: Converged!");
                }
                break;
            }

            if self.debug_level >= 1 && (generation % 10 == 0 || generation == self.num_gens - 1) {
                let first_front_size = individuals.iter().filter(|ind| ind.rank == 1).count();
                println!(
                    "NSGA-II: Gen {} | Best fitness: {:.4} | First front size: {} | Pop size: {}",
                    generation,
                    best_fitness,
                    first_front_size,
                    individuals.len()
                );
            }
        }

        // Extract the Pareto front (first front).
        individuals
            .iter()
            .filter(|ind| ind.rank == 1)
            .cloned()
            .collect()
    }
}

/// To be used when running directly
impl HpMocd {
    pub fn _new(graph: Graph) -> Self {
        HpMocd {
            graph,
            debug_level: 10,
            pop_size: 100,
            num_gens: 100,
            cross_rate: 0.8,
            mut_rate: 0.2,
        }
    }

    pub fn _run(&self) -> Partition {
        let first_front = self.envolve();
        let best_solution = max_q_selection(&first_front);

        normalize_community_ids(best_solution.partition.clone())
    }
}

#[pymethods]
impl HpMocd {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = 0,
        pop_size = 100,
        num_gens = 100,
        cross_rate = 0.8,
        mut_rate = 0.2
    ))]
    pub fn new(
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
    ) -> PyResult<Self> {
        let edges = get_edges(graph)?;
        let graph = build_graph(edges);

        Ok(HpMocd {
            graph,
            debug_level,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
        })
    }

    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let first_front = self.envolve();

        Ok(first_front
            .into_iter()
            .map(|ind| (normalize_community_ids(ind.partition), ind.objectives))
            .collect())
    }

    #[pyo3(signature = ())]
    pub fn run(&self) -> PyResult<Partition> {
        let first_front = self.envolve();
        let best_solution = max_q_selection(&first_front);

        Ok(normalize_community_ids(best_solution.partition.clone()))
    }
}
