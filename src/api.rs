//! api.rs
//! Implements some python-facing APIs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
// ================================================================================================

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::core::graph::Partition;
use crate::core::hpmocd::HpMocd;
use crate::core::hpmocd::{
    DEFAULT_CROSS_RATE as HPMOCD_DEFAULT_CROSS_RATE,
    DEFAULT_DEBUG_LEVEL as HPMOCD_DEFAULT_DEBUG_LEVEL,
    DEFAULT_MUT_RATE as HPMOCD_DEFAULT_MUT_RATE, DEFAULT_NUM_GENS as HPMOCD_DEFAULT_NUM_GENS,
    DEFAULT_POP_SIZE,
};
use crate::core::prism::Prism;
use crate::core::prism::config::{
    DEFAULT_ARCHIVE_CAP, DEFAULT_BETA, DEFAULT_DEBUG_LEVEL, DEFAULT_LPA_FRAC, DEFAULT_LPA_ITERS,
    DEFAULT_MUT_RATE, DEFAULT_NUM_GENS, DEFAULT_POLISH_ITERS, DEFAULT_SWARM_SIZE,
    DEFAULT_TURBULENCE_FRAC, DEFAULT_V_MAX,
};

// ================================================================================================

/// Run PRISM with defaults. For tuning, use the ``Prism`` class.
///
/// Returns ``dict[node, community]``. Isolated nodes get ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "prism", signature = (graph))]
pub fn prism_fn(py: Python<'_>, graph: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let instance = Prism::new(
        py,
        graph,
        DEFAULT_DEBUG_LEVEL,
        DEFAULT_SWARM_SIZE,
        DEFAULT_NUM_GENS,
        DEFAULT_ARCHIVE_CAP,
        DEFAULT_MUT_RATE,
        DEFAULT_TURBULENCE_FRAC,
        DEFAULT_V_MAX,
        DEFAULT_BETA,
        DEFAULT_LPA_FRAC,
        DEFAULT_LPA_ITERS,
    )?;
    instance.run(py, DEFAULT_POLISH_ITERS)
}

/// Run HP-MOCD (NSGA-II) with defaults. For tuning, use the ``HpMocd`` class.
///
/// Returns ``dict[node, community]``. Isolated nodes get ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "hpmocd", signature = (graph))]
pub fn hpmocd_fn(py: Python<'_>, graph: &Bound<'_, PyAny>) -> PyResult<Partition> {
    let instance = HpMocd::new(
        py,
        graph,
        HPMOCD_DEFAULT_DEBUG_LEVEL,
        DEFAULT_POP_SIZE,
        HPMOCD_DEFAULT_NUM_GENS,
        HPMOCD_DEFAULT_CROSS_RATE,
        HPMOCD_DEFAULT_MUT_RATE,
        None,
    )?;
    instance.run(py)
}

