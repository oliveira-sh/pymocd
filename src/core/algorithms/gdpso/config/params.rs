//! The knob struct of one GDPSO run and the sanitising of its rates.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::gdpso::sampling::{coefficient, probability};

use super::defaults::{
    DEFAULT_C1, DEFAULT_C2, DEFAULT_LPA_SWEEPS, DEFAULT_MUT_FRAC, DEFAULT_MUT_RATE,
    DEFAULT_NUM_GENS, DEFAULT_POP_SIZE, DEFAULT_W,
};

/// Every knob of one GDPSO run. The last two fields are not part of the method:
/// they exist so the two documented ablations need no code change.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub pop_size: usize,
    pub num_gens: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
    /// per-node broadcast probability inside a mutated particle
    pub mut_rate: f64,
    /// fraction of the swarm that is mutated
    pub mut_frac: f64,
    pub lpa_sweeps: usize,
    /// true reproduces the reference, which mutates the first
    /// `pop_size * mut_frac` slots every generation; false mutates each
    /// particle with probability `mut_frac`
    pub mut_by_index: bool,
    /// ablation: replace the whole velocity update with this fixed move
    /// probability
    pub const_move_prob: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pop_size: DEFAULT_POP_SIZE,
            num_gens: DEFAULT_NUM_GENS,
            w: DEFAULT_W,
            c1: DEFAULT_C1,
            c2: DEFAULT_C2,
            mut_rate: DEFAULT_MUT_RATE,
            mut_frac: DEFAULT_MUT_FRAC,
            lpa_sweeps: DEFAULT_LPA_SWEEPS,
            mut_by_index: true,
            const_move_prob: None,
        }
    }
}

impl Config {
    // the one choke point: no rate or coefficient reaches an operator unsanitised
    pub(in crate::core::algorithms::gdpso) fn sanitized(self) -> Self {
        Self {
            pop_size: self.pop_size.max(1),
            w: coefficient(self.w),
            c1: coefficient(self.c1),
            c2: coefficient(self.c2),
            mut_rate: probability(self.mut_rate),
            mut_frac: probability(self.mut_frac),
            const_move_prob: self.const_move_prob.map(probability),
            ..self
        }
    }
}
