//! Default parameters for Shi-MOCD (PESA-II).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_RAND_NETWORKS: usize = 30;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
// Shi 2012 Table 1: "pc and pm are 0.6 and 0.4 for these four EA based
// algorithms" — fixed across all of the paper's experiments.
pub const DEFAULT_CROSS_RATE: f64 = 0.6;
pub const DEFAULT_MUT_RATE: f64 = 0.4;

// External-archive (EP) capacity cap: Shi 2012 Table 1 never grows epsize
// past 100; epsize is derived as `min(pop_size, EPSIZE_CAP)`.
pub const EPSIZE_CAP: usize = 100;

// `mocd_q` / `mocd_d` wrapper defaults — the HP-MOCD benchmark budget, so the
// baseline compares apples-to-apples (Santos et al. 2025: C_R=0.9, M_R=0.1).
pub const BENCH_CROSS_RATE: f64 = 0.9;
pub const BENCH_MUT_RATE: f64 = 0.1;
// MOCD-D control fronts (Shi 2012 §3.2 generates three).
pub const MOCD_D_RAND_NETWORKS: usize = 3;
