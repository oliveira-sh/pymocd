//! Leoben Algorithm
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod crossover;
mod individual;
mod mutation;
mod objective;
mod population;
mod utils;

use crate::debug;
use crate::graph::{Graph, Partition};
use crate::leobenalg::objective::calculate_objectives;
use crate::utils::normalize_community_ids;
use individual::{Individual, create_offspring};
use utils::{calculate_crowding_distance, fast_non_dominated_sort, max_q_selection};

use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use std::cmp::Ordering;

const TOURNAMENT_SIZE: usize = 2;

fn evaluate_population(
    individuals: &mut [Individual],
    graph: &Graph,
    // Remove degrees parameter since it's now part of graph
) {
    individuals.par_iter_mut().for_each(|ind| {
        ind.objectives = calculate_objectives(graph, &ind.partition);
    });
}

fn update_population_sort_and_truncate(individuals: &mut Vec<Individual>, pop_size: usize) {
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

fn evolve(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    debug_level: i8,
) -> Vec<Individual> {
    let mut individuals: Vec<Individual> = population::generate_initial_population(graph, pop_size)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    evaluate_population(&mut individuals, graph);

    for generation in 0..num_gens {
        update_population_sort_and_truncate(&mut individuals, pop_size);

        let mut offspring =
            create_offspring(&individuals, graph, cross_rate, mut_rate, TOURNAMENT_SIZE);
        evaluate_population(&mut offspring, graph);

        individuals.extend(offspring);

        if debug_level >= 1 && (generation % 10 == 0 || generation == num_gens - 1) {
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

pub fn print_graph_memory_stats(graph: &Graph) {
    let stats = graph.memory_stats();
    debug!(debug, "Graph Memory Usage:");
    debug!(debug, "  Nodes: {} bytes", stats.nodes_memory);
    debug!(debug, "  Edges: {} bytes", stats.edges_memory);
    debug!(debug, "  Adjacency: {} bytes", stats.adjacency_memory);
    debug!(debug, "  Degrees: {} bytes", stats.degrees_memory);
    debug!(debug,"  Edge Lookup: {} bytes", stats.edge_lookup_memory);
    debug!(warn, "  Total: {} bytes ({:.2} MB)", stats.total(), stats.total() as f64 / 1024.0 / 1024.0);
}

#[pyfunction]
#[pyo3(signature = (graph,
        debug_level = 0,
        pop_size = 100,
        num_gens = 50,
        cross_rate = 0.8,
        mut_rate = 0.5
))]
pub fn leoben(
    graph: &Bound<'_, PyAny>,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> PyResult<Partition> {
    let graph: Graph = Graph::from_python(graph);

    
    if debug_level >= 1 {
        debug!(
            debug,
            "Debug: {} | Level: {}",
            debug_level >= 1,
            debug_level
        );
        graph.print();
    }

    print_graph_memory_stats(&graph);

    let first_front = evolve(
        &graph,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        debug_level,
    );
    let best_solution = max_q_selection(&first_front);

    Ok(normalize_community_ids(
        &graph,
        best_solution.partition.clone(),
    ))
}
