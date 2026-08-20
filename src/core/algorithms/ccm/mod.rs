//! NSGA-III-CCM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021); see README.md.
//!
//! `nsga3` is a self-contained, sequential NSGA-III and deliberately does not
//! reuse this crate's shared engine or Rayon: this baseline's cost and
//! behaviour have to track the paper's.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
mod locus;
mod nsga3;
mod objectives;

#[cfg(test)]
mod fixtures;

pub use api::{ccm, ccm_fronts};
pub use defaults::*;
