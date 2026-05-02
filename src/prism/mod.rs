//! prism module. See README.md for the full algorithm description.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod api;
mod archive;
mod config;
pub(crate) mod dense;
mod eval;
mod evolve;
mod particle;
mod seed;

use crate::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub struct Prism {
    graph: Graph,
    debug_level: i8,
    swarm_size: usize,
    num_gens: usize,
    archive_cap: usize,
    mut_rate: f64,
    turbulence_frac: f64,
    v_max: f64,
    beta: f64,
    lpa_frac: f64,
    lpa_iters: usize,
    on_generation: Option<Py<PyAny>>,
}
