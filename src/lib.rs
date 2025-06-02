//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
mod graph; // graph custom implementation;
mod hpmocd;
mod macros; // debug!, and more or future macros
mod mocd; // deprecated
mod operators; // genetic algorithm operators;
mod utils; // networkx to graph conversion, some useful funcs.
mod xfeats; // extra-features

// ================================================================================================

pub use hpmocd::HpMocd; // proposed hpmocd (2025)
pub use mocd::MOCD; // shi 2010, (with a lot of changes) [deprecated]

use xfeats::{fitness, set_thread_count};

// ================================================================================================

/// pymocd is a Python library, powered by a Rust backend, for performing efficient multi-objective
/// evolutionary community detection in complex networks.
/// This library is designed to deliver enhanced performance compared to traditional methods,
/// making it particularly well-suited for analyzing large-scale graphs.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;
    m.add_class::<HpMocd>()?;
    m.add_class::<MOCD>()?;
    Ok(())
}
