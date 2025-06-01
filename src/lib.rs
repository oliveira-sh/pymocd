//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::ThreadPoolBuilder;
use std::sync::Once;

// ================================================================================================
// Algorithms Classes
// ================================================================================================
mod hpmocd;
mod mocd;

mod graph; // graph custom implementation;
mod operators; // genetic algorithm operators;
mod utils; // networkx to graph conversion, some useful funcs.

// ================================================================================================

static INIT_RAYON: Once = Once::new();
pub use hpmocd::HpMocd; // proposed hpmocd (2025)
// pub use mocd::MOCD; // shi 2010, (with a lot of changes, cant be called the same alg)

// ================================================================================================
// Functions
// ================================================================================================

#[macro_export]
macro_rules! debug {
    // invocation: debug!(debug  , "something {} happened", x);
    //            debug!(warn, "watch out: {}", y);
    //            debug!(err  , "failed: {:?}", err);
    ($level:ident, $($arg:tt)*) => {
        {
            let (lvl, color) = match stringify!($level) {
                "debug"   => ("DEBUG"  , "\x1b[34m"),
                "warn" => ("WARNING", "\x1b[33m"),
                "err"   => ("ERROR"  , "\x1b[31m"),
                other     => (other   , "\x1b[0m"),
            };

            let file = file!();
            let line = line!();
            println!(
                "{}[ {}: {}:{} {}]\x1b[0m {}",
                color, lvl, file, line, color,
                format_args!($($arg)*)
            );
        }
    };
}

/// Calculates the Q score for a given graph and community partition
/// based on (Shi, 2012) multi-objective modularity equation. Q = 1 - intra - inter
///
/// # Parameters
/// - `graph` (networkx.Graph): The graph to analyze
/// - `partition` (dict[int, int]): Dictionary mapping nodes to community IDs
///
/// # Returns
/// - float
#[pyfunction(name = "fitness")]
fn fitness(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let edges = utils::get_edges(graph)?;
    let graph = utils::build_graph(edges);

    Ok(operators::get_modularity_from_partition(
        &utils::to_partition(partition)?,
        &graph,
    ))
}

#[pyfunction]
fn set_thread_count(num_threads: usize) -> PyResult<()> {
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
// ================================================================================================
// Module
// ================================================================================================

#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;
    m.add_class::<HpMocd>()?;
    //m.add_class::<MOCD>()?;
    Ok(())
}
