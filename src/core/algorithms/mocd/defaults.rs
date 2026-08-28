//! Default parameters for Shi-MOCD (PESA-II).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_RAND_NETWORKS: usize = 3;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.6;
pub const DEFAULT_MUT_RATE: f64 = 0.4;
pub const EPSIZE_CAP: usize = 100;

pub const BENCH_CROSS_RATE: f64 = 0.9;
pub const BENCH_MUT_RATE: f64 = 0.1;
pub const MOCD_D_RAND_NETWORKS: usize = 3;
