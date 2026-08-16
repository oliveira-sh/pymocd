//! NSGA-II primitives: non-dominated sorting, crowding distance and the
//! crowding-based environment selection shared by both swarms.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod crowding;
mod sorting;
mod survival;

pub type Obj = Vec<f64>;

pub(super) use crowding::crowding_distance;
pub(super) use sorting::fast_nondominated_sort;
pub(super) use survival::environment_selection;
