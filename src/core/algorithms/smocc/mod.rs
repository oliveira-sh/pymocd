//! SMOCC — Sparse Multi-Objective Co-evolutionary Community detection,
//! a macro-micro co-evolutionary detector (Zhang et al., IEEE CIM lineage)
//! reformulated over a CSR graph and a sparse edge similarity for
//! near-linear memory/time.
//!
//! The paper's Louvain local-search step is deliberately deleted, not
//! feature-flagged: re-adding it is a change of algorithm and must be
//! re-measured. Shipped defaults are the measured winners; see `config::defaults`.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod config;
mod front;
mod macro_micro;
mod nsga2;
mod objectives;
mod operators;
mod similarity;
mod utils;

pub type Labels = Vec<i32>;
pub type Genome = Vec<u8>;

pub use api::{smocc, smocc_fronts};
pub use config::defaults::*;
