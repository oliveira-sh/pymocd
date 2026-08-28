//! NSGA-III-KRM (Shaik, Ravi & Deb 2021) module root; see README.md.
//!
//! Sequential and self-contained on purpose — no shared engine, no Rayon — so
//! this baseline's cost tracks the published method.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod locus;
mod nsga3;
mod objectives;
mod operators;

#[cfg(test)]
mod fixtures;

pub use api::{krm, krm_fronts};
pub use defaults::*;
