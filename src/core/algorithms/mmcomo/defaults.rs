//! Shipped defaults for MMCoMO (paper Table II / Section IV-A).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 50;
pub const DEFAULT_CROSS_RATE: f64 = 0.1;
pub const DEFAULT_MUT_RATE: f64 = 0.1;
pub const DEFAULT_GAP: usize = 10;
/// Never specified in the paper; reimplementation choice, exposed for tuning.
pub const DEFAULT_BETA: f64 = 0.05;
