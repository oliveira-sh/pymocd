//! The MOGA-Net evolutionary core.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod individual;

pub(super) use engine::run;
pub(super) use individual::{Individual, fast_non_dominated_sort};
