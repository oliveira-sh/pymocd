//! Locus-respecting genetic operators (Pizzuti 2009, Sec. 4).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::{Genome, Locus};
use rand::{Rng, RngExt}; // rand 0.10: random_range/random_bool live on RngExt

/// Uniform crossover (Pizzuti 2009, Sec. 4): "a random binary vector is
/// created. Uniform crossover then selects the genes where the vector is 1
/// from the first parent, and the genes where the vector is 0 from the
/// second parent... Thus from two safe parents a safe child is generated."
/// Per-gene independent coin flip; no repair needed since every gene value
/// is copied verbatim from one of the two already-safe parents.
pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
    a.iter()
        .zip(b.iter())
        .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
        .collect()
}

/// Repaired mutation (Pizzuti 2009, Sec. 4): "the possible values an allele
/// can assume are restricted to the neighbors of gene i. This repaired
/// mutation guarantees the generation of a safe mutated child in which each
/// node is linked only with one of its neighbors." Each gene independently,
/// with probability `mut_rate`, is resampled uniformly from `{node itself} ∪
/// neighbours(node)` -- this can never produce an invalid (unsafe) locus
/// value.
pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            let cands = &locus.candidates[p];
            *gene = cands[rng.random_range(0..cands.len())];
        }
    }
}
