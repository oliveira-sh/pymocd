//! Shipped defaults for NSGA-III-KRM; see README.md.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

// Must stay at or above the reference-point count H = C(M + p - 1, p) = 91,
// for M = 3 objectives and p = DEFAULT_DIVISIONS.
pub const DEFAULT_POP_SIZE: usize = 100;

pub const DEFAULT_NUM_GENS: usize = 100;

pub const DEFAULT_CROSS_RATE: f64 = 0.8;

pub const DEFAULT_MUT_RATE: f64 = 1.0 / 34.0;

pub const DEFAULT_DIVISIONS: usize = 12;
