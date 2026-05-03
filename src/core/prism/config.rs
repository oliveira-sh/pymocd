//! Tunable defaults and small math helpers.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CommunityId;
use crate::core::prism::archive::Archive;
use crate::core::prism::dense::DensePartition;
use rand::RngExt;

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_SWARM_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_ARCHIVE_CAP: usize = 100;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
pub const DEFAULT_TURBULENCE_FRAC: f64 = 0.15;
pub const DEFAULT_POLISH_ITERS: usize = 20;

pub const DEFAULT_INERTIA_LOW: f64 = 0.1;
pub const DEFAULT_INERTIA_HIGH: f64 = 0.4;
pub const DEFAULT_V_MAX: f64 = 0.3;
pub const DEFAULT_PHI_SCALE: f64 = 0.1;
pub const DEFAULT_INITIAL_V: f64 = 0.1;
pub const DEFAULT_BETA: f64 = 0.7;
pub const DEFAULT_LPA_FRAC: f64 = 0.7;
pub const DEFAULT_LPA_ITERS: usize = 3;

pub const LOCAL_OPT_EVERY: usize = 5;
pub const LOCAL_OPT_TOP_N: usize = 10;
pub const LOCAL_OPT_ITERS: usize = 2;

fn constriction(phi: f64) -> f64 {
    if phi <= 4.0 {
        return 1.0;
    }
    let disc = (phi * phi - 4.0 * phi).sqrt();
    2.0 / (2.0 - phi - disc).abs()
}

pub fn adaptive_chi(archive: &Archive, generation: usize, total_gens: usize) -> f64 {
    let t = if total_gens <= 1 {
        1.0
    } else {
        generation as f64 / (total_gens as f64 - 1.0)
    };
    let phi = 4.05 + 0.15 * t;
    let base = constriction(phi);
    let diverse = archive.len() >= 3;
    if diverse { base } else { base * 0.9 }
}

pub fn random_dense_partition(n: usize) -> DensePartition {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| rng.random_range(0..n as CommunityId))
        .collect()
}

pub fn perturb_partition(p: &[CommunityId], frac: f64) -> DensePartition {
    let mut rng = rand::rng();
    let n = p.len();
    let mut out = p.to_vec();
    for j in 0..n {
        if rng.random::<f64>() < frac {
            out[j] = rng.random_range(0..n as CommunityId);
        }
    }
    out
}
