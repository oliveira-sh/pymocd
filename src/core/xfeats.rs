//! xfeats.rs
//! Implements extra features for the library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::ThreadPoolBuilder;
use std::sync::Once;

use crate::core::graph::Graph;
use crate::core::operators;
use crate::core::utils;

static INIT_RAYON: Once = Once::new();

/// Modularity Q (Shi, 2012): ``Q = 1 - intra - inter``.
#[gen_stub_pyfunction]
#[pyfunction(name = "fitness")]
pub fn fitness(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let graph = Graph::from_python(graph);

    Ok(operators::get_modularity_from_partition(
        &utils::to_partition(partition)?,
        &graph,
    ))
}
/// Set Rayon's global thread count. First call wins; later calls are no-ops.
#[gen_stub_pyfunction]
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
