//! MOPSO — multi-objective particle swarm optimisation over the Constant Potts
//! Model.
//!
//! CPM is `H(gamma) = sum_c [e_c - gamma C(n_c,2)]`. Splitting it the way
//! HP-MOCD splits modularity gives two minimised objectives, an edge term and a
//! pair term, whose weighted sums are exactly the `gamma` family. So the Pareto
//! front of this pair is the graph's whole resolution profile, CPM's resolution
//! parameter stops being a parameter, and the swarm's archive is the answer at
//! every scale at once.
//!
//! Each particle is niched at one resolution and its local move optimises CPM
//! there, which is what spreads the swarm along the profile instead of piling
//! it onto one granularity. Its objective stays resolution-free, so the archive
//! remains a single global front.
//!
//! Deterministic: one RNG stream per (iteration, particle), no shared state
//! inside a flight, and the archive is fed in slot order. Same graph and
//! parameters, same partition, on any number of threads.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod config;
mod front;
mod objectives;
mod pareto;
pub mod swarm;
mod utils;

/// A partition, as one community label per dense vertex id. Labels are vertex
/// ids in `[0, n)`, never densified during the search: keeping them in that
/// range is what lets every per-community counter be a flat array.
pub type Labels = Vec<i32>;

pub use api::{Profile, mopso, mopso_fronts};
pub use config::defaults::*;

#[cfg(test)]
pub(crate) fn front_test_select(
    g: &crate::core::graph::CsrGraph,
    front: &[Labels],
    objs: &[objectives::Obj],
    mode: u8,
) -> usize {
    front::select_index_mode(g, front, objs, mode)
}
