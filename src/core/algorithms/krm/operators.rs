//! Locus-respecting genetic operators + NSGA-III binary tournament mating
//! selection. NSGA-III has no crowding-distance concept (that is an
//! NSGA-II-only tie-breaker), so mating here compares `rank` only.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::Individual;
use super::locus::{Genome, Locus};
use rand::{Rng, RngExt}; // rand 0.10: random_range/random_bool live on RngExt
use std::cmp::Ordering;

/// Binary tournament: lower rank wins; a tie (including comparing an
/// individual against itself) is broken by a coin flip. Mirrors pymoo's
/// default NSGA-III mating selection.
pub fn binary_tournament(pop: &[Individual], rng: &mut impl Rng) -> usize {
    let n = pop.len();
    let i = rng.random_range(0..n);
    let j = rng.random_range(0..n);
    match pop[i].rank.cmp(&pop[j].rank) {
        Ordering::Less => i,
        Ordering::Greater => j,
        Ordering::Equal => {
            if rng.random_bool(0.5) {
                i
            } else {
                j
            }
        }
    }
}

/// Uniform, locus-respecting crossover. The paper's text on this point is
/// terse ("customized NSGA-III ... standard crossover"), so this is our exact
/// reading: with probability `cross_rate` the child takes each gene
/// independently from parent A or parent B (50/50 per gene -- classic uniform
/// crossover); otherwise (probability `1 - cross_rate`) the child is a
/// verbatim clone of one parent chosen uniformly at random. Both branches
/// trivially respect the locus constraint, since every gene value is copied
/// unmodified from a valid parent gene at the same position.
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
/// `mut_rate`, is resampled uniformly from `{node itself} ∪ {its
/// neighbours}` -- this can never produce an invalid locus value.
pub fn mutate(genome: &mut Genome, locus: &Locus, mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            let cands = &locus.candidates[p];
            *gene = cands[rng.random_range(0..cands.len())];
        }
    }
}
