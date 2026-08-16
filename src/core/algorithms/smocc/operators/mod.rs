//! Variation operators: crossover and mutation for the macro genomes and the
//! micro label vectors.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod macro_genome;
mod micro_labels;

pub(super) use macro_genome::macro_offspring;
pub(super) use micro_labels::{micro_offspring, micro_offspring_topo};
