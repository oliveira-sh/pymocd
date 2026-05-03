//! High-Perfomance Multiobjective community detection
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod individual;
mod utils;

use crate::core::graph::{Graph, Partition};
use crate::core::utils::normalize_community_ids;
use crate::core::operators;
use crate::debug;
use individual::{Individual, create_offspring};
use utils::{calculate_crowding_distance, fast_non_dominated_sort, max_q_selection};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::cmp::Ordering;
use std::collections::HashMap;

const TOURNAMENT_SIZE: usize = 2;

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.7;
pub const DEFAULT_MUT_RATE: f64 = 0.5;

/// NSGA-II multi-objective community detection.
///
/// Args:
///     graph: networkx.Graph or DiGraph.
///     debug_level: 0 silent, 1+ logs every 10 generations.
///     pop_size: NSGA-II population size.
///     num_gens: number of generations.
///     cross_rate: crossover probability.
///     mut_rate: mutation probability.
///     objectives: optional list of callables ``(graph, partition) -> float``
///         to minimise; replaces the built-in intra/inter objectives.
#[gen_stub_pyclass]
#[pyclass]
pub struct HpMocd {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    /// Python graph object, stored so Python objectives can receive it.
    py_graph: Option<Py<PyAny>>,
    /// Python callables `(graph, partition: dict) -> float` to minimise.
    /// When non-empty, replaces the default Rust intra/inter objectives.
    py_objectives: Vec<Py<PyAny>>,
    /// Optional Python callable invoked after every generation:
    ///   on_generation(generation: int, num_gens: int, front_size: int) -> None
    on_generation: Option<Py<PyAny>>,
}

/* Private (Not exposed to py user) */
impl HpMocd {
    fn evaluate_population(
        &self,
        py: Option<Python<'_>>,
        individuals: &mut [Individual],
        graph: &Graph,
        degrees: &HashMap<i32, usize, FxBuildHasher>,
    ) -> PyResult<()> {
        if self.py_objectives.is_empty() {
            individuals.par_iter_mut().for_each(|ind| {
                let metrics = operators::get_fitness(graph, &ind.partition, degrees, true);
                ind.objectives = vec![metrics.intra, metrics.inter];
            });
        } else {
            // Python objectives path: sequential (GIL-bound)
            let py = py.expect("Python token required when py_objectives are set");
            let py_graph = self
                .py_graph
                .as_ref()
                .expect("py_graph must be set when py_objectives are used");
            let py_objs = &self.py_objectives;

            // Allocate one dict and reuse it across all individuals.
            // Objectives must not store a reference to the dict between calls.
            let partition_dict = PyDict::new(py);
            let graph_ref = py_graph.bind(py);
            for ind in individuals.iter_mut() {
                partition_dict.clear();
                for (&node, &comm) in ind.partition.iter() {
                    partition_dict.set_item(node, comm)?;
                }
                let mut objectives = Vec::with_capacity(py_objs.len());
                for obj in py_objs.iter() {
                    let value = obj
                        .bind(py)
                        .call1((graph_ref, &partition_dict))?
                        .extract::<f64>()?;
                    objectives.push(value);
                }
                ind.objectives = objectives;
            }
        }
        Ok(())
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

    fn envolve(&self, py: Option<Python<'_>>) -> PyResult<Vec<Individual>> {
        let degrees = &self.graph.precompute_degrees();
        let mut individuals: Vec<Individual> =
            operators::generate_population(&self.graph, self.pop_size)
                .into_par_iter()
                .map(Individual::new)
                .collect();
        self.evaluate_population(py, &mut individuals, &self.graph, degrees)?;

        for generation in 0..self.num_gens {
            self.update_population_sort_and_truncate(&mut individuals, self.pop_size);

            let mut offspring = create_offspring(
                &individuals,
                &self.graph,
                self.cross_rate,
                self.mut_rate,
                TOURNAMENT_SIZE,
            );
            self.evaluate_population(py, &mut offspring, &self.graph, degrees)?;

            individuals.extend(offspring);

            let first_front_size = individuals.iter().filter(|ind| ind.rank == 1).count();

            if self.debug_level >= 1 && (generation % 10 == 0 || generation == self.num_gens - 1) {
                debug!(
                    debug,
                    "NSGA-II: Gen {} | 1st Front/Pop: {}/{}",
                    generation,
                    first_front_size,
                    individuals.len()
                );
            }

            if let Some(cb) = &self.on_generation {
                if let Some(py) = py {
                    cb.bind(py)
                        .call1((generation, self.num_gens, first_front_size))?;
                }
            }
        }

        Ok(individuals
            .iter()
            .filter(|ind| ind.rank == 1)
            .cloned()
            .collect())
    }
}

/// To be used when running directly (no Python objectives)
impl HpMocd {
    pub fn _new(graph: Graph) -> Self {
        HpMocd {
            graph,
            debug_level: 10,
            pop_size: 100,
            num_gens: 100,
            cross_rate: 0.8,
            mut_rate: 0.2,
            py_graph: None,
            py_objectives: vec![],
            on_generation: None,
        }
    }

    pub fn _run(&self) -> Partition {
        let first_front = self.envolve(None).expect("envolve failed");
        let best_solution = max_q_selection(&first_front);

        normalize_community_ids(&self.graph, best_solution.partition.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl HpMocd {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = 0,
        pop_size = 100,
        num_gens = 100,
        cross_rate = 0.7,
        mut_rate = 0.5,
        objectives = None
    ))]
    pub fn new(
        _py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
        objectives: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python(graph);

        if debug_level >= 1 {
            debug!(
                debug,
                "Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            rust_graph.print();
        }

        // Always store the Python graph so set_objectives() works after construction.
        let py_graph = Some(graph.clone().unbind());
        let py_objectives: Vec<Py<PyAny>> = objectives
            .map(|obj_list| obj_list.iter().map(|item| item.unbind()).collect())
            .unwrap_or_default();

        Ok(HpMocd {
            graph: rust_graph,
            debug_level,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
            py_graph,
            py_objectives,
            on_generation: None,
        })
    }

    /// Replace the objectives. Empty list reverts to built-in intra/inter.
    #[pyo3(signature = (objectives))]
    pub fn set_objectives(&mut self, objectives: &Bound<'_, PyList>) -> PyResult<()> {
        self.py_objectives = objectives.iter().map(|item| item.unbind()).collect();
        Ok(())
    }

    /// Register per-generation callback ``(gen, num_gens, front_size) -> None``.
    /// Pass ``None`` to clear.
    #[pyo3(signature = (callback))]
    pub fn set_on_generation(&mut self, callback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.on_generation = callback.map(|cb| cb.clone().unbind());
        Ok(())
    }

    /// Configured number of generations.
    #[getter]
    pub fn num_gens(&self) -> usize {
        self.num_gens
    }

    /// Return all non-dominated solutions as ``[(partition, objectives), ...]``.
    /// Objective order matches the configured objectives (or intra/inter).
    #[pyo3(signature = ())]
    pub fn generate_pareto_front(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let first_front = self.envolve(Some(py))?;

        Ok(first_front
            .into_iter()
            .map(|ind| {
                (
                    normalize_community_ids(&self.graph, ind.partition),
                    ind.objectives,
                )
            })
            .collect())
    }

    /// Run and return the best partition (max-Q from the Pareto front).
    /// Isolated nodes get community ``-1``.
    #[pyo3(signature = ())]
    pub fn run(&self, py: Python<'_>) -> PyResult<Partition> {
        let first_front: Vec<Individual> = self.envolve(Some(py))?;
        let best_solution: &Individual = max_q_selection(&first_front);

        Ok(normalize_community_ids(
            &self.graph,
            best_solution.partition.clone(),
        ))
    }
}
