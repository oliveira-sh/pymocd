//! Pareto machinery: dominance, crowding, and the bounded external archive.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod archive;
mod crowding;
mod dominance;

pub(super) use archive::Archive;
#[cfg(test)]
pub(super) use dominance::dominates;
