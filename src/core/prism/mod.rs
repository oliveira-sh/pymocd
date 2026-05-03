//! prism module. See README.md for the full algorithm description.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod api;
mod archive;
pub(crate) mod config;
pub(crate) mod dense;
mod eval;
mod evolve;
mod particle;
mod seed;

use crate::core::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Particle-swarm multi-objective community detection.
///
/// Args:
///     graph: networkx.Graph or DiGraph.
///     debug_level: 0 silent, 1+ logs progress.
///     swarm_size: particles per generation.
///     num_gens: number of generations.
///     archive_cap: max Pareto archive size.
///     mut_rate: mutation probability.
///     turbulence_frac: fraction perturbed during turbulence.
///     v_max: velocity clamp.
///     beta: inertia/cognitive blend.
///     lpa_frac: fraction of swarm seeded via LPA.
///     lpa_iters: LPA iterations during seeding.
#[gen_stub_pyclass]
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
