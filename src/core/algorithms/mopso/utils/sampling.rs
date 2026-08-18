//! The determinism contract: one independent RNG stream per (iteration, particle).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

const RNG_BASE: u64 = 0x5A17_71C1_E5EE_D0F1;

/// The RNG a particle uses at one iteration, `salt` naming the iteration and `slot` the
/// particle. No stream is shared, so the result cannot depend on how rayon schedules them.
pub fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

/// A uniform draw in `[0, 1)`, the primitive every stochastic decision here is built from.
#[inline(always)]
pub fn unit(r: &mut impl Rng) -> f64 {
    r.random::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_reproducible_and_independent() {
        let take = |salt, slot| {
            let mut r = slot_rng(salt, slot);
            (0..8).map(|_| unit(&mut r)).collect::<Vec<f64>>()
        };
        assert_eq!(take(1, 0), take(1, 0), "a stream is not reproducible");
        assert_ne!(take(1, 0), take(1, 1), "two slots share a stream");
        assert_ne!(take(1, 0), take(2, 0), "two iterations share a stream");
    }

    #[test]
    fn unit_stays_in_range() {
        let mut r = slot_rng(7, 3);
        for _ in 0..4096 {
            let u = unit(&mut r);
            assert!((0.0..1.0).contains(&u), "{u} outside [0,1)");
        }
    }
}
