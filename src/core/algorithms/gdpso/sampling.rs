//! The deterministic per-slot RNG contract for gdpso: one stream per slot.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::SeedableRng;
use rand::rngs::StdRng;

// deliberately duplicated across smocc, mopots and gdpso: one seeding
// contract stated per module, no cross-module dependency
const RNG_BASE: u64 = 0x5CA1_E5EED;

/// One independent stream per `(salt, slot)` pair; never depends on the thread count.
pub fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

/// Clamps a rate into `[0.0, 1.0]`, mapping non-finite input to `0.0`.
#[inline]
pub const fn probability(p: f64) -> f64 {
    if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 }
}

/// Clamps a PSO coefficient to a finite non-negative value: a negative or nan
/// `w`/`c1`/`c2` would push the sigmoid argument outside the `[0, 3.722]` band
/// the method assumes.
#[inline]
pub const fn coefficient(c: f64) -> f64 {
    if c.is_finite() && c > 0.0 { c } else { 0.0 }
}
