//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
mod csicea_m;
mod graph;
mod hpmocd;
mod macros;
mod mocd; // deprecated
mod operators;
mod utils;
mod xfeats; // extra-features

// ================================================================================================

// proposed csicea (2026)
use csicea_m::csicea;

// proposed hpmocd (2025)
use hpmocd::HpMocd;

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
    m.add_function(wrap_pyfunction!(csicea, m)?)?;
    m.add_class::<HpMocd>()?;
    Ok(())
}
