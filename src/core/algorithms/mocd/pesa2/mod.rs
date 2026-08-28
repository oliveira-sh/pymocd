//! Single-threaded PESA-II (Corne, Jerram, Knowles & Oates 2001), the search
//! Shi et al. 2012 §3 drives. See ../README.md.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod archive;
mod engine;
mod grid;
mod solution;

pub(super) use engine::evolutionary_phase;
pub(super) use solution::Solution;
