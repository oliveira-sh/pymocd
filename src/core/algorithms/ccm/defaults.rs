//! Shipped defaults for NSGA-III-CCM (Shaik, Ravi & Deb 2021, Table 3).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

// 200, not the '20' of Sec. 8.1: Table 3 and Fig. S.22 use 200, and pop_size
// has to stay above the reference-point count H = 91 (M=3, divisions=12).
pub const DEFAULT_POP_SIZE: usize = 200;

pub const DEFAULT_NUM_GENS: usize = 100;

pub const DEFAULT_CROSS_RATE: f64 = 0.8;

pub const DEFAULT_MUT_RATE: f64 = 1.0 / 68.0;

pub const DEFAULT_DIVISIONS: usize = 12;

pub const DEFAULT_R: f64 = 1.0;

pub const DEFAULT_ALPHA: f64 = 1.0;
