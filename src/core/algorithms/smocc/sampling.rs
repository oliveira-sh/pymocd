//! The deterministic RNG contract: one independent stream per population slot,
//! plus the draws every variation operator shares.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::distr::Bernoulli;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

const RNG_BASE: u64 = 0x5CA1_E5EED;

pub(super) fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

#[inline]
pub(super) fn bernoulli(p: f64) -> Bernoulli {
    match Bernoulli::new(p) {
        Ok(d) => d,
        Err(_) => panic!("p={p:?} is outside range [0.0, 1.0]"),
    }
}

#[inline]
pub(super) fn tournament(ranks: &[usize], crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = ranks.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    if ranks[i] < ranks[j] || (ranks[i] == ranks[j] && crowd[i] >= crowd[j]) {
        i
    } else {
        j
    }
}
