//! MO-POTS: NSGA-II over the exact (cut, pair) split of the Constant Potts Model.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod front;
mod nsga2;
mod objectives;
mod operators;
mod sampling;

#[cfg(test)]
mod fixtures;

pub use api::MoPots;
pub use defaults::*;

// smocc's CsrGraph cpm() is cross-checked against this hashmap implementation
#[cfg(test)]
pub(crate) use objectives::calculate_objectives;
