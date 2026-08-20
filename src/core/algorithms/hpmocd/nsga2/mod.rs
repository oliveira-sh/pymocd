//! NSGA-II for HP-MOCD: population member, ranking, crowding survival, generational loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod individual;
mod offspring;
mod sorting;
mod survival;

pub(super) use engine::evolve;
pub(super) use individual::{Individual, TOURNAMENT_SIZE};
