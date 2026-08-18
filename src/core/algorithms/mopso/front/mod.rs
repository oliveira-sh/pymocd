//! The label-free choice of a single member out of the archive's resolution profile.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod modularity;
mod plateau;
mod select;

pub(super) use select::{select_best, select_index};
