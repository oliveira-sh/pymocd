//! HP-MOCD: NSGA-II over Shi's decomposed modularity, on a label-map encoding.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod nsga2;
mod objectives;
mod operators;
mod select;

pub use api::HpMocd;
pub use defaults::*;
