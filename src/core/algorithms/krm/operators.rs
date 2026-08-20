//! Locus-respecting genetic operators and mating selection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};

use super::locus::{Genome, Locus};

/// Uniform draw, not a tournament: pymoo's NSGA-III mating comparison ranks
/// only constraint violation, so unconstrained it always falls through to a
/// coin flip.
pub fn random_parent(pop_len: usize, rng: &mut impl Rng) -> usize {
    rng.random_range(0..pop_len)
}

/// Both branches keep the locus constraint: every gene is copied from the same
/// position of a parent.
pub fn crossover(a: &Genome, b: &Genome, cross_rate: f64, rng: &mut impl Rng) -> Genome {
    if rng.random_bool(cross_rate) {
        a.iter()
            .zip(b.iter())
            .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
            .collect()
    } else if rng.random_bool(0.5) {
        a.clone()
    } else {
        b.clone()
    }
}

pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            let cands = &locus.candidates[p];
            *gene = cands[rng.random_range(0..cands.len())];
        }
    }
}
