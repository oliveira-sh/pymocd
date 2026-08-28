//! HP-MOCD pyclass: the Python surface, objective evaluation and the search entry points.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

use crate::core::graph::normalize_community_ids;
use crate::core::graph::{Graph, NodeId, Partition};

use super::defaults::{
    DEFAULT_CROSS_RATE, DEFAULT_DEBUG_LEVEL, DEFAULT_MUT_RATE, DEFAULT_NUM_GENS, DEFAULT_POP_SIZE,
};
use super::nsga2::{Individual, TOURNAMENT_SIZE, evolve};
use super::objectives::calculate_objectives;
use super::select::max_q_selection;

type Degrees = HashMap<NodeId, usize, FxBuildHasher>;

const LOG_EVERY_GENERATIONS: usize = 10;

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
#[pyclass]
pub struct HpMocd {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    py_graph: Option<Py<PyAny>>,
    py_objectives: Vec<Py<PyAny>>,
    on_generation: Option<Py<PyAny>>,
}

impl HpMocd {
    fn evaluate_with_rust_objectives(individuals: &mut [Individual], graph: &Graph, degrees: &Degrees) {
        individuals.par_iter_mut().for_each(|ind| {
            let metrics = calculate_objectives(graph, &ind.partition, degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
        });
    }

    fn evaluate_with_py_objectives(
        &self,
        py: Python<'_>,
        individuals: &mut [Individual],
    ) -> PyResult<()> {
        let py_graph = self
            .py_graph
            .as_ref()
            .expect("py_graph must be set when py_objectives are used");
        let graph_ref = py_graph.bind(py);

        // one dict is allocated and cleared per individual, so an objective must not
        // hold on to it between calls
        let partition_dict = PyDict::new(py);
        for ind in individuals.iter_mut() {
            partition_dict.clear();
            for (&node, &comm) in &ind.partition {
                partition_dict.set_item(node, comm)?;
            }
            let mut objectives = Vec::with_capacity(self.py_objectives.len());
            for obj in &self.py_objectives {
                let value = obj
                    .bind(py)
                    .call1((graph_ref, &partition_dict))?
                    .extract::<f64>()?;
                objectives.push(value);
            }
            ind.objectives = objectives;
        }
        Ok(())
    }

    fn evaluate_population(
        &self,
        py: Option<Python<'_>>,
        individuals: &mut [Individual],
        graph: &Graph,
        degrees: &Degrees,
    ) -> PyResult<()> {
        if self.py_objectives.is_empty() {
            Self::evaluate_with_rust_objectives(individuals, graph, degrees);
            Ok(())
        } else {
            let py = py.expect("Python token required when py_objectives are set");
            self.evaluate_with_py_objectives(py, individuals)
        }
    }

    fn report_generation(
        &self,
        py: Option<Python<'_>>,
        generation: usize,
        num_gens: usize,
        population: &[Individual],
    ) -> PyResult<()> {
        let first_front_size = population.iter().filter(|ind| ind.rank == 1).count();

        if self.debug_level >= 1
            && (generation.is_multiple_of(LOG_EVERY_GENERATIONS) || generation == num_gens - 1)
        {
            debug!(
                debug,
                "NSGA-II: Gen {} | 1st Front/Pop: {}/{}",
                generation,
                first_front_size,
                population.len()
            );
        }

        if let Some(cb) = &self.on_generation
            && let Some(py) = py
        {
            cb.bind(py)
                .call1((generation, num_gens, first_front_size))?;
        }
        Ok(())
    }

    fn search(&self, py: Option<Python<'_>>) -> PyResult<Vec<Individual>> {
        let degrees = self.graph.precompute_degrees();

        let individuals = evolve(
            &self.graph,
            self.pop_size,
            self.num_gens,
            self.cross_rate,
            self.mut_rate,
            TOURNAMENT_SIZE,
            |inds| self.evaluate_population(py, inds, &self.graph, degrees),
            |generation, num_gens, pop| self.report_generation(py, generation, num_gens, pop),
        )?;

        Ok(individuals
            .into_iter()
            .filter(|ind| ind.rank == 1)
            .collect())
    }
}

#[pymethods]
impl HpMocd {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = DEFAULT_DEBUG_LEVEL,
        pop_size = DEFAULT_POP_SIZE,
        num_gens = DEFAULT_NUM_GENS,
        cross_rate = DEFAULT_CROSS_RATE,
        mut_rate = DEFAULT_MUT_RATE,
        objectives = None
    ))]
    #[allow(clippy::too_many_arguments)] // pyo3 constructor: one arg per Python kwarg
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

        // stored unconditionally: set_objectives() may add objectives after construction
        let py_graph = Some(graph.clone().unbind());
        let py_objectives: Vec<Py<PyAny>> = objectives
            .map(|obj_list| obj_list.iter().map(pyo3::Bound::unbind).collect())
            .unwrap_or_default();

        Ok(Self {
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
        self.py_objectives = objectives.iter().map(pyo3::Bound::unbind).collect();
        Ok(())
    }

    /// Register per-generation callback ``(gen, num_gens, front_size) -> None``.
    /// Pass ``None`` to clear.
    #[pyo3(signature = (callback))]
    pub fn set_on_generation(&mut self, callback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.on_generation = callback.map(|cb| cb.clone().unbind());
        Ok(())
    }

    #[getter]
    pub const fn num_gens(&self) -> usize {
        self.num_gens
    }

    /// Return all non-dominated solutions as ``[(partition, objectives), ...]``.
    /// Objective order matches the configured objectives (or intra/inter).
    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self, py: Python<'_>) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let first_front = self.search(Some(py))?;

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
        let first_front: Vec<Individual> = self.search(Some(py))?;
        let best_solution: &Individual = max_q_selection(&first_front);

        Ok(normalize_community_ids(
            &self.graph,
            best_solution.partition.clone(),
        ))
    }
}
