//! What happens to the Pareto front once the search ends: union refinement and
//! the label-free choice of a single member.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod agglom;
mod components;
mod refine;
mod select;
mod tiny;

pub(super) use refine::{refine_front, refine_front_mode};
pub(super) use select::{select_best, select_index, select_index_mode};
