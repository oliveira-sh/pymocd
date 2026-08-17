//! The macro-micro co-evolution: the two swarms, their periodic exchange and
//! the generational loop that drives them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod exchange;
mod init;
mod swarms;

pub(super) use engine::run_fronts;
pub(super) use init::{init_macro_genomes, init_micro_labels, macro_cmax};
