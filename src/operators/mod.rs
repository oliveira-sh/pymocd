//! Evolutionary operators for multiobjective community detection
//! 
//! This module provides the core evolutionary operators used in genetic algorithms
//! for community detection, including crossover, mutation, selection, and population
//! generation operators.

use crate::graph::{Graph, Partition};
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

pub mod crossover;
pub mod mutation;
pub mod objectives;
pub mod population;
pub mod selection;

// Re-export commonly used types
pub use objectives::{calculate_objectives, calculate_modularity, validate_metrics};
pub use crossover::{two_point_crossover, ensemble_crossover};
pub use mutation::mutate;
pub use population::generate_initial_population;
pub use selection::{tournament_selection, fast_non_dominated_sort, calculate_crowding_distance};

// Import convergence criteria from utils
pub use crate::utils::ConvergenceCriteria;

/// Convenience functions for backward compatibility

/// Performs crossover between two parent partitions
/// 
/// # Arguments
/// * `parent1` - First parent partition
/// * `parent2` - Second parent partition  
/// * `crossover_rate` - Probability of crossover occurring
/// 
/// # Returns
/// Offspring partition created from crossover
pub fn crossover(parent1: &Partition, parent2: &Partition, crossover_rate: f64) -> Partition {
    two_point_crossover(parent1, parent2, crossover_rate)
}

/// Performs mutation on a partition
/// 
/// # Arguments
/// * `partition` - Partition to mutate (modified in-place)
/// * `graph` - Graph structure for neighborhood information
/// * `mutation_rate` - Probability of each node being mutated
pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    mutate(partition, graph, mutation_rate);
}

/// Calculates fitness metrics for a partition
/// 
/// # Arguments
/// * `graph` - The graph being analyzed
/// * `partition` - The partition to evaluate
/// * `degrees` - Precomputed node degrees for efficiency
/// * `parallel` - Whether to use parallel computation
/// 
/// # Returns
/// Metrics containing fitness values
pub fn get_fitness(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
    parallel: bool,
) -> objectives::Metrics {
    calculate_objectives(graph, partition, degrees, parallel)
}

/// Generates an initial population of partitions
/// 
/// # Arguments
/// * `graph` - The graph to create partitions for
/// * `population_size` - Number of partitions to generate
/// 
/// # Returns
/// Vector of randomly generated partitions
pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    generate_initial_population(graph, population_size)
}

/// Calculates modularity for a partition
/// 
/// # Arguments
/// * `partition` - The partition to evaluate
/// * `graph` - The graph being analyzed
/// 
/// # Returns
/// Modularity value
pub fn get_modularity_from_partition(partition: &Partition, graph: &Graph) -> f64 {
    let metrics = calculate_objectives(graph, partition, &graph.precompute_degrees(), false);
    metrics.modularity()
}
