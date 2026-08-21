//! MOGA-Net (Pizzuti, IEEE ICTAI 2009 / IEEE TEC 16(3):418-430, 2012): a
//! self-contained, single-threaded reimplementation maximizing (Community
//! Score, Community Fitness) over a locus-based genome. See `README.md`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod locus;
mod objectives;
mod operators;
mod search;

pub use api::{moga_net, moga_net_fronts};
pub use defaults::*;
