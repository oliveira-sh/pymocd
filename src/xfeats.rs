//! xfeats.rs
//! Implements extra features for the library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::ThreadPoolBuilder;
use std::sync::Once;

use crate::debug;
use crate::operators;
use crate::utils;

static INIT_RAYON: Once = Once::new();

/// Calculates the Q score for a given graph and community partitio
/// based on (Shi, 2012) multi-objective modularity equation. Q = 1 - intra - inter
///
/// Parameters
/// graph: (networkx.Graph) - The graph to analyze
/// partition: (dict[int, int]) - Dictionary mapping nodes to community IDs
///
/// Returns
/// float: f64
#[pyfunction(name = "fitness")]
pub fn fitness(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let edges = utils::get_edges(graph)?;
    let graph = utils::build_graph(edges);

    Ok(operators::get_modularity_from_partition(
        &utils::to_partition(partition)?,
        &graph,
    ))
}
/// Setup the maximum amount of logical/virtual threads rayon should use
///
/// Parameters
/// num_threads: usize - The number of threads you wish to use [0, inf]
#[pyfunction]
pub fn set_thread_count(num_threads: usize) -> PyResult<()> {
    INIT_RAYON.call_once(|| {
        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
        debug!(warn, "Global thread pool initialized initialized with {} threads", num_threads);
        debug!(warn, "Using set_thread_count again has no effect, due to static ThreadPoolBuilder initialization")
    });
    Ok(())
}
