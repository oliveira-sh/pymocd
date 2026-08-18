//! Tunable configuration for MOPSO: the parameter bundle the swarm reads.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod defaults;

use defaults::{
    DEFAULT_COGNITIVE, DEFAULT_INERTIA, DEFAULT_LOCAL_RATE, DEFAULT_NUM_GENS, DEFAULT_POP_SIZE,
    DEFAULT_SEED_ROUNDS, DEFAULT_SOCIAL,
};

/// Everything the swarm needs that is not the graph.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub pop: usize,
    pub gens: usize,
    pub inertia: f64,
    pub cognitive: f64,
    pub social: f64,
    pub local_rate: f64,
    pub archive: usize,
    pub seed_rounds: usize,
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
            archive: DEFAULT_POP_SIZE, // one slot per particle: the archive holds the whole profile.
            seed_rounds: DEFAULT_SEED_ROUNDS,
        }
    }
}

impl Cfg {
    /// Out-of-range values are clamped rather than rejected, because these arrive from Python.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pop: usize,
        gens: usize,
        inertia: f64,
        cognitive: f64,
        social: f64,
        local_rate: f64,
        archive: usize,
        seed_rounds: usize,
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
            seed_rounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_parameters_are_clamped_not_trusted() {
        let c = Cfg::new(0, 10, -1.0, 5.0, f64::NAN, 0.5, 0, 2);
        assert_eq!(c.pop, 2);
        assert_eq!(c.archive, 2);
        assert_eq!(c.inertia, 0.0);
        assert_eq!(c.cognitive, 1.0);
        assert_eq!(c.social, 0.0);
        assert_eq!(c.local_rate, 0.5);
    }

    #[test]
    fn the_default_bundle_matches_the_shipped_constants() {
        let d = Cfg::default();
        assert_eq!(d.pop, DEFAULT_POP_SIZE);
        assert_eq!(d.archive, DEFAULT_POP_SIZE);
        assert_eq!(d.local_rate, DEFAULT_LOCAL_RATE);
    }
}
