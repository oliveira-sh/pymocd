//! Default parameters for GDPSO (Cai et al., Inf. Sci. 316:503-516, 2015).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 250;

pub const DEFAULT_W: f64 = 0.7298;
pub const DEFAULT_C1: f64 = 1.4961;
pub const DEFAULT_C2: f64 = 1.4961;

pub const DEFAULT_MUT_RATE: f64 = 0.1;
pub const DEFAULT_MUT_FRAC: f64 = 0.1;

pub const DEFAULT_LPA_SWEEPS: usize = 5;
