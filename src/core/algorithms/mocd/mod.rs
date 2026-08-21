//! Shi-MOCD (Shi, Yan, Cai & Wu 2012): PESA-II over a locus-based genome and
//! decomposed-modularity objectives, with both of the paper's model selectors.
//! See `README.md`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod locus;
mod null_model;
mod objectives;
mod operators;
mod pesa2;
mod selection;

pub use api::Mocd;
pub use defaults::*;
