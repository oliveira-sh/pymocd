//! Variation operators: mating, the two offspring generators and the local search.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod local_search;
mod macro_genome;
mod mating;
mod micro_labels;

pub(super) use local_search::local_search;
pub(super) use macro_genome::macro_offspring;
pub(super) use micro_labels::micro_offspring;
