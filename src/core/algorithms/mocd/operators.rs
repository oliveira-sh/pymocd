//! The variation operators over the locus genome.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;

use super::locus::{Genome, NodeIndex};

/// Shi's "uniform two-point crossover" — by the paper's own functional
/// description, plain per-gene uniform crossover. Each allele is inherited at
/// its own position, so a child of two legal genomes is always legal.
pub fn uniform_crossover(p1: &Genome, p2: &Genome, rng: &mut impl rand::Rng) -> Genome {
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
        .collect()
}

/// Per-gene adjacency mutation with independent probability `p_m`. The
/// resample is a uniform draw over the whole candidate set, so it may return
/// the current allele.
pub fn mutate(genome: &mut Genome, idx: &NodeIndex, p_m: f64, rng: &mut impl rand::Rng) {
    for (gene, cands) in genome.iter_mut().zip(&idx.neighbor_candidates) {
        if rng.random_bool(p_m) {
            *gene = cands[rng.random_range(0..cands.len())];
        }
    }
}
