//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::redundant_pub_crate,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::missing_errors_doc,
    clippy::unreadable_literal
)]

mod api;
mod config;
mod gpu;
mod mopso;
mod objectives;
mod refine;
mod select;
mod sim;

#[cfg(test)]
mod tests;

pub type Labels = Vec<i32>;
pub type Genome = Vec<u8>;

pub use api::{gmocs, gmocs_fronts};
pub use config::defaults::*;
