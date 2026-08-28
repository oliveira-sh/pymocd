//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod defaults;

use defaults::{
    DEFAULT_COGNITIVE, DEFAULT_INERTIA, DEFAULT_LOCAL_RATE, DEFAULT_LS_PERIOD, DEFAULT_NUM_GENS,
    DEFAULT_POP_SIZE, DEFAULT_SOCIAL,
};

#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub pop: usize,
    pub gens: usize,
    pub inertia: f64,
    pub cognitive: f64,
    pub social: f64,
    pub local_rate: f64,
    pub archive: usize,
    pub ls_period: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            pop: DEFAULT_POP_SIZE,
            gens: DEFAULT_NUM_GENS,
            inertia: DEFAULT_INERTIA,
            cognitive: DEFAULT_COGNITIVE,
            social: DEFAULT_SOCIAL,
            local_rate: DEFAULT_LOCAL_RATE,
            archive: DEFAULT_POP_SIZE,
            ls_period: DEFAULT_LS_PERIOD,
        }
    }
}

impl Cfg {
    pub fn new(
        pop: usize,
        gens: usize,
        inertia: f64,
        cognitive: f64,
        social: f64,
        local_rate: f64,
        archive: usize,
    ) -> Self {
        let rate = |x: f64| if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        Self {
            pop: pop.max(2),
            gens,
            inertia: rate(inertia),
            cognitive: rate(cognitive),
            social: rate(social),
            local_rate: rate(local_rate),
            archive: archive.max(2),
            ls_period: DEFAULT_LS_PERIOD,
        }
    }
}
