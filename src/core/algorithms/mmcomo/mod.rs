//! MMCoMO: macro-micro co-evolutionary multi-objective community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod defaults;
#[cfg(test)]
mod fixtures;
mod graph;
mod macro_micro;
mod nsga2;
mod objectives;
mod operators;
mod similarity;

/// `n x n` diffusion-kernel similarity matrix.
pub type Sm = Vec<Vec<f64>>;
/// Micro representation: one community label per node index. Every label is
/// itself a node index in `[0, n)`, which the objectives, the local search and
/// the vote matrix rely on to index their accumulators by label directly.
pub type Labels = Vec<i32>;
/// Macro representation: one centre bit per node index.
pub type Genome = Vec<u8>;

pub use api::{mmcomo, mmcomo_fronts};
pub use defaults::*;
pub use graph::Graph;
