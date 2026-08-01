//! Locus-respecting genetic operators (Pizzuti 2009, Sec. 4).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::{Genome, Locus};
use rand::{Rng, RngExt}; // rand 0.10: random_bool lives on RngExt

/// Uniform crossover (Pizzuti 2009, Sec. 4): per-gene independent coin flip;
/// no repair needed since every gene value is copied verbatim from one of
/// the two already-safe parents.
pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
    a.iter()
        .zip(b.iter())
        .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
        .collect()
}

/// Repaired mutation (Pizzuti 2009, Sec. 4): each gene independently, with
/// probability `mut_rate`, is resampled uniformly from neighbours(node) --
/// "restricted to the neighbors of gene i"; self only for isolated nodes.
pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            *gene = locus.random_allele(p, rng);
        }
    }
}
