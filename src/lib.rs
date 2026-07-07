//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
use pyo3::prelude::*;
mod api;
pub mod core;
use api::{
    ccm_fn, hpmocd_fn, krm_fn, mmcomo_fn, mmcomo_fronts_fn, moga_net_fn, mocd_d_fn, mocd_q_fn,
    dcsbm_full_mdl_fn, dcsbm_mdl_fn, gt_metrics_fn, scale_fn, scale_fronts_fn, scale_fronts_raw_fn, sbm_mdl_fn,
};
use core::algorithms::hpmocd::HpMocd;
use core::algorithms::mocd::Mocd;
use core::xfeats::{fitness, set_thread_count};

/// pymocd is a Python library, powered by a Rust backend, for performing efficient community
/// detection in complex networks.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;
    m.add_function(wrap_pyfunction!(hpmocd_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_q_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_d_fn, m)?)?;
    m.add_function(wrap_pyfunction!(moga_net_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ccm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(krm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mmcomo_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mmcomo_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(scale_fn, m)?)?;
    m.add_function(wrap_pyfunction!(scale_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(scale_fronts_raw_fn, m)?)?;
    m.add_function(wrap_pyfunction!(sbm_mdl_fn, m)?)?;
    m.add_function(wrap_pyfunction!(dcsbm_mdl_fn, m)?)?;
    m.add_function(wrap_pyfunction!(dcsbm_full_mdl_fn, m)?)?;
    m.add_function(wrap_pyfunction!(gt_metrics_fn, m)?)?;
    m.add_class::<HpMocd>()?;
    m.add_class::<Mocd>()?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
