//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::defaults::{W_MAX, W_MIN};

pub(crate) fn inertia(t: usize, num_gens: usize) -> f64 {
    if num_gens <= 1 {
        return W_MIN;
    }
    W_MAX - (W_MAX - W_MIN) * (t - 1) as f64 / (num_gens - 1) as f64
}

pub(crate) fn turbulence(turb: f64, w: f64) -> f64 {
    (turb * w / W_MAX).clamp(0.0, 1.0)
}
