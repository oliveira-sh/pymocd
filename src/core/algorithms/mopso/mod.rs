//! MOPSO: multi-objective particle swarm optimisation over the Constant Potts Model.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod config;
mod front;
mod objectives;
mod pareto;
mod swarm;
mod utils;

pub type Labels = Vec<i32>; // one community label per dense vertex id, always in [0, n).

pub use api::{Profile, mopso, mopso_fronts};
pub use config::defaults::*;
