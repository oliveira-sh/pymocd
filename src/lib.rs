//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
// ================================================================================================
use pyo3::prelude::*;
mod api;
pub mod core;
// ================================================================================================
use api::{hpmocd_fn, prism_fn};
use core::hpmocd::HpMocd;
use core::prism::Prism;
use core::xfeats::{fitness, set_thread_count};
// ================================================================================================

/// pymocd is a Python library, powered by a Rust backend, for performing efficient community
/// detection in complex networks.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;
    m.add_function(wrap_pyfunction!(prism_fn, m)?)?;
    m.add_function(wrap_pyfunction!(hpmocd_fn, m)?)?;
    m.add_class::<HpMocd>()?;
    m.add_class::<Prism>()?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
