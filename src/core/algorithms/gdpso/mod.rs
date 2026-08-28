//! GDPSO — Greedy Discrete Particle Swarm Optimization (Cai, Gong, Ma, Ruan,
//! Yuan, Jiao, Information Sciences 316:503-516, 2015). Newman-Girvan
//! modularity is the single maximised objective, so there is no Pareto front
//! and no `gdpso_fronts` entry point. The method, its parameters and its
//! divergences from the reference are documented in `README.md`.
//!
//! Written from a clean-room description of the authors' reference
//! implementation; no reference source was read while writing this module.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod config;
mod local;
mod objective;
mod sampling;
mod swarm;

pub use api::{gdpso, gdpso_on_graph, gdpso_with};
pub use config::Config;
pub use config::defaults::*;
