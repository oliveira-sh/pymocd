//! Python module entry point for the pymocd library.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
mod api;
mod core;
use api::detectors::*;
use api::max_cores;
use api::metrics::*;

/// Python Multi-objective Community Detection (pymocd) is a Python library, powered by
/// a Rust backend, for performing efficient community detection in complex networks.
/// Get your graph, call a method, and we'll offer you a community.
/// Recommended Methods: `scale` or `hpmocd`.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // utils
    m.add_function(wrap_pyfunction!(max_cores, m)?)?;

    // detectors -> partition
    m.add_function(wrap_pyfunction!(hpmocd_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_q_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_d_fn, m)?)?;
    m.add_function(wrap_pyfunction!(moga_net_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ccm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(krm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mmcomo_fn, m)?)?;
    m.add_function(wrap_pyfunction!(scale_fn, m)?)?;

    // detectors -> pareto frontier
    m.add_function(wrap_pyfunction!(mmcomo_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(scale_fronts_fn, m)?)?;

    // evaluation metrics
    m.add_function(wrap_pyfunction!(gt_metrics_fn, m)?)?;
    m.add_function(wrap_pyfunction!(nmi_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ami_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ari_fn, m)?)?;
    m.add_function(wrap_pyfunction!(f1_fn, m)?)?;

    // -- finished --
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
