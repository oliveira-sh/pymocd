//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos.

pub const TOURNAMENT_SIZE: usize = 2;
pub const ENSEMBLE_SIZE: usize = 4;
pub const N_OBJ: usize = 2;
pub const MIGRATION_GENS: usize = 200;
pub const DEFAULT_POP_SIZE: usize = 200;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
pub const DEFAULT_GAMMA: f64 = 1.0;
pub const DEFAULT_SEED: u64 = 42;

pub const CONV_PVAL: f64 = 0.05;
pub const CHECK_EVERY: usize = 10;
pub const MIN_GENS: usize = 20;
pub const STOP_WARMUP: usize = 20;
pub const STOP_WINDOW: usize = 10;
pub const STOP_ALPHA: f64 = 0.05;
