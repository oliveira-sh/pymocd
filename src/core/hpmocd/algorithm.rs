//! Main HP-MOCD algorithm implementation
//! 
//! This module implements the High-Performance Multiobjective Community Detection algorithm
//! using NSGA-II for multiobjective optimization.

use crate::graph::{Graph, Partition};
use crate::utils::normalize_community_ids;
use crate::{debug, operators};
use super::individual::{Individual, create_offspring};
use super::selection::{calculate_crowding_distance, fast_non_dominated_sort, max_q_selection};

use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::cmp::Ordering;
use std::collections::HashMap;

const TOURNAMENT_SIZE: usize = 2;

/// High-Performance Multiobjective Community Detection algorithm
/// 
/// This struct implements the HP-MOCD algorithm using NSGA-II for multiobjective
/// optimization of community detection objectives.
#[pyclass]
pub struct HpMocd {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
}

/* Private implementation (not exposed to Python users) */
impl HpMocd {
    /// Evaluates the fitness of all individuals in the population
    fn evaluate_population(
        &self,
        individuals: &mut [Individual],
        graph: &Graph,
        degrees: &HashMap<i32, usize, FxBuildHasher>,
    ) {
        individuals.par_iter_mut().for_each(|ind| {
            let metrics = operators::get_fitness(graph, &ind.partition, degrees, true);
            ind.objectives = [metrics.intra, metrics.inter];
        });
    }

    /// Updates population by sorting and truncating to desired size
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

    /// Main evolutionary algorithm loop
    fn evolve(&self) -> Vec<Individual> {
        let degrees = &self.graph.precompute_degrees();
        let mut individuals: Vec<Individual> =
            operators::generate_population(&self.graph, self.pop_size)
                .into_par_iter()
                .map(Individual::new)
                .collect();
        self.evaluate_population(&mut individuals, &self.graph, degrees);

        for generation in 0..self.num_gens {
            self.update_population_sort_and_truncate(&mut individuals, self.pop_size);

            let mut offspring = create_offspring(
                &individuals,
                &self.graph,
                self.cross_rate,
                self.mut_rate,
                TOURNAMENT_SIZE,
            );
            self.evaluate_population(&mut offspring, &self.graph, degrees);

            individuals.extend(offspring);

            if self.debug_level >= 1 && (generation % 10 == 0 || generation == self.num_gens - 1) {
                let first_front_size = individuals.iter().filter(|ind| ind.rank == 1).count();
                debug!(
                    debug,
                    "NSGA-II: Gen {} | 1st Front/Pop: {}/{}",
                    generation,
                    first_front_size,
                    individuals.len()
                );
            }
        }

        individuals
            .iter()
            .filter(|ind| ind.rank == 1)
            .cloned()
            .collect()
    }
}

/// Internal constructor for direct Rust usage
impl HpMocd {
    /// Creates a new HP-MOCD instance for direct Rust usage
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

    /// Runs the algorithm and returns the best partition
    pub fn _run(&self) -> Partition {
        let first_front = self.evolve();
        let best_solution = max_q_selection(&first_front);

        normalize_community_ids(&self.graph, best_solution.partition.clone())
    }
}

/// Python interface methods
#[pymethods]
impl HpMocd {
    /// Creates a new HP-MOCD instance from Python
    /// 
    /// # Arguments
    /// * `graph` - Python graph object (NetworkX or igraph)
    /// * `debug_level` - Debug output level (0=none, higher=more verbose)
    /// * `pop_size` - Population size for genetic algorithm
    /// * `num_gens` - Number of generations to run
    /// * `cross_rate` - Crossover probability
    /// * `mut_rate` - Mutation probability
    /// 
    /// # Returns
    /// New HpMocd instance
    #[new]
    #[pyo3(signature = (graph,
        debug_level = 0,
        pop_size = 100,
        num_gens = 100,
        cross_rate = 0.7,
        mut_rate = 0.5
    ))]
    pub fn new(
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
    ) -> PyResult<Self> {
        let graph = Graph::from_python(graph);

        if debug_level >= 1 {
            debug!(
                debug,
                "Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            graph.print();
        }

        Ok(HpMocd {
            graph,
            debug_level,
            pop_size,
            num_gens,
            cross_rate: cross_rate,
            mut_rate,
        })
    }

    /// Generates the Pareto front of solutions
    /// 
    /// Returns all non-dominated solutions found during the optimization process.
    /// Each solution includes the partition and its objective values.
    /// 
    /// # Returns
    /// Vector of (partition, objectives) tuples representing the Pareto front
    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self) -> PyResult<Vec<(Partition, [f64; 2])>> {
        let first_front = self.evolve();

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

    /// Runs the HP-MOCD algorithm and returns the best partition
    /// 
    /// Executes the multiobjective evolutionary algorithm and selects the best
    /// solution based on modularity (Q-value).
    ///
    /// # Returns
    /// Dictionary mapping node IDs to community IDs. Isolated nodes (degree 0)
    /// are assigned community ID -1.
    #[pyo3(signature = ())]
    pub fn run(&self) -> PyResult<Partition> {
        let first_front = self.evolve();
        let best_solution = max_q_selection(&first_front);

        Ok(normalize_community_ids(
            &self.graph,
            best_solution.partition.clone(),
        ))
    }

    /// Gets algorithm configuration as a string
    pub fn get_config(&self) -> String {
        format!(
            "HpMocd(pop_size={}, num_gens={}, cross_rate={:.2}, mut_rate={:.2}, debug_level={})",
            self.pop_size, self.num_gens, self.cross_rate, self.mut_rate, self.debug_level
        )
    }

    /// Returns the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Returns the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    /// Returns graph statistics as a dictionary
    pub fn graph_stats(&self) -> PyResult<std::collections::HashMap<String, f64>> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("nodes".to_string(), self.graph.num_nodes() as f64);
        stats.insert("edges".to_string(), self.graph.num_edges() as f64);
        stats.insert("max_degree".to_string(), self.graph.max_degree() as f64);
        stats.insert("avg_degree".to_string(), self.graph.avg_degree());
        Ok(stats)
    }
}