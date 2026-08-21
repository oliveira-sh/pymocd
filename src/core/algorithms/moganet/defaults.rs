//! Shipped MOGA-Net defaults, as recommended by Pizzuti.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_POP_SIZE: usize = 300;
pub const DEFAULT_NUM_GENS: usize = 30;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
// TEVC 2012 Sec. VI-C: "the parameter r ... has been set to 2". Not derivable
// from the paper's reported karate NMI, which her own binary does not produce;
// see README, Parameters.
pub const DEFAULT_R: f64 = 2.0;
pub const DEFAULT_ALPHA: f64 = 1.0;
