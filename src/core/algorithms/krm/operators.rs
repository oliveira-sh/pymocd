//! Locus-respecting genetic operators + uniform-random mating selection
//! (pymoo's NSGA-III tournament degenerates to random picks on unconstrained
//! problems, so the draw is implemented directly).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::{Genome, Locus};
use rand::{Rng, RngExt}; // rand 0.10: random_range/random_bool live on RngExt

/// Uniform-random parent pick. pymoo's NSGA-III mating "tournament" compares
/// only constraint violation, so for unconstrained problems every comparison
/// ties and falls through to a coin flip — i.e. a uniform random draw.
pub fn random_parent(pop_len: usize, rng: &mut impl Rng) -> usize {
    rng.random_range(0..pop_len)
}

/// Uniform, locus-respecting crossover: with probability `cross_rate` each
/// gene comes from parent A or B (50/50 per gene); otherwise the child clones
/// one parent chosen at random. Both branches respect the locus constraint,
/// since every gene is copied from a valid parent gene at the same position.
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

/// Adjacency-constrained mutation: each gene independently, with probability
/// `mut_rate`, is resampled uniformly from `{position itself} ∪ neighbours`.
pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            let cands = &locus.candidates[p];
            *gene = cands[rng.random_range(0..cands.len())];
        }
    }
}
