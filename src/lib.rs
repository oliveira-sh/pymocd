//! PyMOCD - High-Performance Multiobjective Community Detection
//!
//! PyMOCD is a Python library, powered by a Rust backend, for performing efficient
//! community detection in complex networks using multiobjective evolutionary algorithms.
//!
//! # Features
//!
//! - **High Performance**: Optimized Rust implementation with parallel processing
//! - **Multiobjective**: Uses NSGA-II for true multiobjective optimization
//! - **Python Integration**: Compatible with NetworkX and igraph
//! - **Professional Library**: Well-structured, documented, and tested codebase
//!
//! # Architecture
//!
//! The library is organized into several key modules:
//!
//! - [`graph`]: Graph data structures and utilities
//! - [`core`]: Core algorithm implementations (HP-MOCD)
//! - [`operators`]: Evolutionary operators (crossover, mutation, selection)
//! - [`utils`]: Utility functions and helpers
//!
//! # Example Usage
//!
//! ```python
//! import networkx as nx
//! import pymocd
//!
//! # Create a graph
//! G = nx.karate_club_graph()
//!
//! # Run community detection
//! algorithm = pymocd.HpMocd(G)
//! partition = algorithm.run()
//!
//! # Get Pareto front
//! pareto_front = algorithm.generate_pareto_front()
//! ```

use pyo3::prelude::*;

// Core modules
pub mod core;
pub mod graph;
pub mod operators;
pub mod utils;

// Testing (only in test builds)
#[cfg(test)]
pub mod tests;

// Re-export main types for easier access
pub use core::{AlgorithmConfig, HpMocd};
pub use graph::{CommunityId, Graph, NodeId, Partition};
pub use operators::{ConvergenceCriteria, objectives::Metrics};
pub use utils::get_thread_count;

// Import the main algorithm and utility functions for Python binding
use utils::set_thread_count as set_thread_count_impl;

/// Sets the number of threads for parallel processing
///
/// This function is exposed to Python to allow control over the number of threads
/// used by the parallel algorithms.
///
/// # Arguments
/// * `num_threads` - Number of threads to use (0 for automatic detection)
///
/// # Returns
/// The actual number of threads that will be used
#[pyfunction]
pub fn set_thread_count(num_threads: usize) -> usize {
    set_thread_count_impl(num_threads)
}

/// Calculates fitness metrics for a given partition
///
/// This function is exposed to Python for standalone fitness evaluation.
///
/// # Arguments
/// * `graph` - Python graph object (NetworkX or igraph)
/// * `partition` - Python dictionary with node->community mappings
/// * `parallel` - Whether to use parallel computation
///
/// # Returns
/// Tuple of (intra, inter) objective values
#[pyfunction]
pub fn fitness(
    graph: &Bound<'_, PyAny>,
    partition: &Bound<'_, pyo3::types::PyDict>,
    parallel: Option<bool>,
) -> PyResult<(f64, f64)> {
    let graph = Graph::from_python(graph);
    let partition = utils::to_partition(partition)?;
    let degrees = graph.precompute_degrees().clone();
    let use_parallel = parallel.unwrap_or(false);

    let metrics = operators::calculate_objectives(&graph, &partition, &degrees, use_parallel);
    Ok((metrics.intra, metrics.inter))
}

/// PyMOCD Python module
///
/// This function defines the Python module interface, exposing the main
/// algorithm class and utility functions to Python users.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add utility functions
    m.add_function(wrap_pyfunction!(set_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;

    // Add main algorithm class
    m.add_class::<HpMocd>()?;

    Ok(())
}

// Library-level documentation and examples
#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        // Test that main types are properly exported
        let _config = AlgorithmConfig::default();
        let _graph = Graph::new();

        // Test utility functions
        let thread_count = set_thread_count(4);
        assert!(thread_count > 0);
    }
}
