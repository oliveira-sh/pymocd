//! The deterministic per-slot RNG contract for mopots: one stream per slot.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::SeedableRng;
use rand::distr::Bernoulli;
use rand::rngs::StdRng;

// shared with smocc so both modules document one seeding contract
const RNG_BASE: u64 = 0x5CA1_E5EED;

/// One independent stream per `(salt, slot)` pair; never depends on the thread count.
pub fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

/// A coin with probability `p`, rejecting anything outside `[0.0, 1.0]`.
#[inline]
pub fn bernoulli(p: f64) -> Bernoulli {
    match Bernoulli::new(p) {
        Ok(d) => d,
        Err(e) => panic!("p={p:?} is outside range [0.0, 1.0]: {e:?}"),
    }
}
