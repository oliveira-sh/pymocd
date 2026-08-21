//! Locus-respecting genetic operators (Pizzuti 2009, Sec. 4).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};

use super::locus::{Genome, Locus};

pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
    a.iter()
        .zip(b.iter())
        .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
        .collect()
}

/// Per-gene mutation, "restricted to the neighbors of gene i" (Pizzuti 2009,
/// Sec. 4), which is what keeps every gene safe without a repair pass.
pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            *gene = locus.random_allele(p, rng);
        }
    }
}
