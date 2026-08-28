//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

const RNG_BASE: u64 = 0x5A17_71C1_E5EE_D0F1;

pub fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

#[inline(always)]
pub fn unit(r: &mut impl Rng) -> f64 {
    r.random::<f64>()
}
