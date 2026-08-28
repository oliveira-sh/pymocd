//! The MOGA-Net generational loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
//!
//! Two deliberate divergences from the textbook reading of the paper, both
//! UNVERIFIABLE: Pizzuti's reference implementation (Moganet2016) ships
//! `MOGANet.p` / `create_population.p` / `crossover_cluster.p` /
//! `mutate_cluster.p` as obfuscated MATLAB P-code, so neither can be settled
//! against the source. The objectives, by contrast, ARE verified against her
//! binary (see `README.md`); only the search loop is in question.
//!   1. Replacement. The paper describes NSGA-II, whose replacement combines
//!      parents and offspring into a 2N pool and truncates by (rank, crowding).
//!      This loop instead copies the top `elite_count` and fills the rest with
//!      fresh children, so a child never competes against the parent generation.
//!   2. Mutation. The paper applies mutation to crossover offspring; here every
//!      child is mutated, including one that was cloned rather than crossed
//!      (`crossoverFraction` in `gamultiobj` can be read either way, and the
//!      composed reading is the one kept).

use rand::rngs::ThreadRng;
use rand::{RngExt, rng};
use std::cmp::Ordering;

use crate::core::algorithms::moganet::locus::{Genome, Locus};
use crate::core::algorithms::moganet::objectives::community_objectives;
use crate::core::algorithms::moganet::operators::{crossover, mutate};

use super::individual::{Individual, calculate_crowding_distance, fast_non_dominated_sort};

/// Pizzuti's "elite reproduction 10% of the population size".
const ELITE_FRACTION: f64 = 0.10;

fn make_individual(locus: &Locus, genome: Genome, r: f64, alpha: f64) -> Individual {
    let labels = locus.decode(&genome);
    let (cs, cf) = community_objectives(locus, &labels, r, alpha);
    Individual {
        genome,
        labels,
        objectives: vec![-cs, -cf],
        rank: usize::MAX,
        crowding_distance: 0.0,
    }
}

/// MATLAB `gamultiobj` rank fitness scaling; the paper only says "roulette
/// selection function".
fn rank_fitness_weights(count: usize) -> Vec<f64> {
    (0..count)
        .map(|pos| 1.0 / ((pos + 1) as f64).sqrt())
        .collect()
}

fn roulette_pick(order: &[usize], fitness: &[f64], total: f64, rng: &mut ThreadRng) -> usize {
    let mut x = rng.random::<f64>() * total;
    for (pos, &idx) in order.iter().enumerate() {
        let w = fitness[pos];
        if x <= w {
            return idx;
        }
        x -= w;
    }
    *order.last().unwrap()
}

fn rank_then_crowding_order(pop: &[Individual]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pop.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        pop[a].rank.cmp(&pop[b].rank).then_with(|| {
            pop[b]
                .crowding_distance
                .partial_cmp(&pop[a].crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });
    order
}

/// Returns the final population with ranks assigned; rank-1 selection and the
/// decision rule are the caller's.
pub fn run(
    locus: &Locus,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> Vec<Individual> {
    let mut rng = rng();

    let mut pop: Vec<Individual> = (0..pop_size)
        .map(|_| make_individual(locus, locus.random_genome(&mut rng), r, alpha))
        .collect();

    let elite_count = ((ELITE_FRACTION * pop_size as f64).round() as usize)
        .max(1)
        .min(pop_size);

    for _gen in 0..num_gens {
        fast_non_dominated_sort(&mut pop);
        calculate_crowding_distance(&mut pop);

        let order = rank_then_crowding_order(&pop);
        let fitness = rank_fitness_weights(order.len());
        let total: f64 = fitness.iter().sum();

        let mut next_gen: Vec<Individual> = Vec::with_capacity(pop_size);
        for &idx in order.iter().take(elite_count) {
            next_gen.push(pop[idx].clone());
        }

        while next_gen.len() < pop_size {
            let pa = roulette_pick(&order, &fitness, total, &mut rng);
            let pb = roulette_pick(&order, &fitness, total, &mut rng);
            let mut child_genome = if rng.random_bool(cross_rate) {
                crossover(&pop[pa].genome, &pop[pb].genome, &mut rng)
            } else if rng.random_bool(0.5) {
                pop[pa].genome.clone()
            } else {
                pop[pb].genome.clone()
            };
            mutate(&mut child_genome, locus, mut_rate, &mut rng);
            next_gen.push(make_individual(locus, child_genome, r, alpha));
        }

        pop = next_gen;
    }

    fast_non_dominated_sort(&mut pop);
    pop
}
