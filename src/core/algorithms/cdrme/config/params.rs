//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::defaults::{
    DEFAULT_ALPHA_MUT, DEFAULT_ALPHA_WALK, DEFAULT_ELITE_SIZE, DEFAULT_MUT_SWEEPS, DEFAULT_N_WALK,
    DEFAULT_POP_SIZE,
};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub alpha_walk: f64,
    pub n_walk: usize,
    pub pop_size: usize,
    pub elite_size: usize,
    pub alpha_mut: f64,
    pub mut_sweeps: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            alpha_walk: DEFAULT_ALPHA_WALK,
            n_walk: DEFAULT_N_WALK,
            pop_size: DEFAULT_POP_SIZE,
            elite_size: DEFAULT_ELITE_SIZE,
            alpha_mut: DEFAULT_ALPHA_MUT,
            mut_sweeps: DEFAULT_MUT_SWEEPS,
        }
    }
}

impl Config {
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            alpha_walk: if self.alpha_walk.is_finite() {
                self.alpha_walk.clamp(0.0, 2.0)
            } else {
                DEFAULT_ALPHA_WALK
            },
            n_walk: self.n_walk.max(1),
            pop_size: self.pop_size.max(1),
            elite_size: self.elite_size.clamp(1, self.pop_size.max(1)),
            alpha_mut: if self.alpha_mut.is_finite() {
                self.alpha_mut
            } else {
                DEFAULT_ALPHA_MUT
            },
            mut_sweeps: self.mut_sweeps,
        }
    }
}
