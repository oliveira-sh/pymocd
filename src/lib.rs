//! Python module entry point for the pymocd library.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::decimal_bitwise_operands)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::float_cmp)]
#![allow(clippy::if_not_else)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::inline_always)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::similar_names)]
#![allow(clippy::single_match_else)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::too_long_first_doc_paragraph)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::unused_self)]
#![allow(clippy::while_float)]
#![allow(clippy::wildcard_imports)]

use pyo3::prelude::*;
mod api;
pub mod core;
use api::detectors::*;
use api::max_cores;
use api::metrics::*;

/// Python Multi-objective Community Detection (pymocd) is a Python library, powered by
/// a Rust backend, for performing efficient community detection in complex networks.
/// Get your graph, call a method, and we'll offer you a community.
/// Recommended Methods: `smocc` or `hpmocd`.
#[pymodule]
#[pyo3(name = "pymocd")]
fn pymocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // utils
    m.add_function(wrap_pyfunction!(max_cores, m)?)?;

    // detectors -> partition
    m.add_function(wrap_pyfunction!(hpmocd_fn, m)?)?;
    m.add_function(wrap_pyfunction!(hpmocd_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_q_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mocd_d_fn, m)?)?;
    m.add_function(wrap_pyfunction!(moga_net_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ccm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(krm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mmcomo_fn, m)?)?;
    m.add_function(wrap_pyfunction!(smocc_fn, m)?)?;

    // detectors -> pareto frontier
    m.add_function(wrap_pyfunction!(ccm_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(krm_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(moga_net_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mmcomo_fronts_fn, m)?)?;
    m.add_function(wrap_pyfunction!(smocc_fronts_fn, m)?)?;

    // deprecated aliases (pre-rename API)
    m.add("scale", m.getattr("smocc")?)?;
    m.add("scale_fronts", m.getattr("smocc_fronts")?)?;

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
