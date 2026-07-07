//! Self-contained, single-threaded MOGA-Net generational loop (Pizzuti 2009,
//! Sec. 3-5), reproducing MATLAB's `gamultiobj`-style elitist + roulette
//! generational replacement -- *not* textbook tournament-select +
//! combine-parents-and-offspring-then-truncate NSGA-II. Every generation:
//! rank + crowd the current population, copy the top 10% ("elite
//! reproduction") unchanged into the next generation, then fill the rest by
//! roulette-wheel (fitness-proportionate) selection over the *whole* current
//! population -- never a 2x-sized combined pool.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::{Individual, calculate_crowding_distance, fast_non_dominated_sort};
use super::locus::{Genome, Locus};
use super::operators::{crossover, mutate};
use crate::core::graph::Graph;
use crate::core::metaheuristics::helpers::objectives::community_score_fitness::community_objectives;
use rand::rngs::ThreadRng;
use rand::{RngExt, rng}; // rand 0.10: random_range/random_bool live on RngExt
use std::cmp::Ordering;

/// Decode + evaluate a freshly-built genome (single call -- no batch/parallel
/// evaluation anywhere in this module). Objectives are `[-CS, -CF]` (both
/// maximized in the paper, so both negated for a minimizing dominance rule).
fn make_individual(graph: &Graph, locus: &Locus, genome: Genome, r: f64, alpha: f64) -> Individual {
    let partition = locus.decode(&genome);
    let (cs, cf) = community_objectives(graph, &partition, r, alpha);
    Individual {
        genome,
        partition,
        objectives: vec![-cs, -cf],
        rank: usize::MAX,
        crowding_distance: 0.0,
    }
}

/// Fitness-proportionate (roulette) pick over `order` (population indices
/// sorted by (rank asc, crowding desc)) using precomputed per-position
/// weights `fitness[pos] = 1/sqrt(pos+1)`. Linear scan, consistent with this
/// repo's other roulette implementations.
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

/// Generational loop: init (safe random genomes) -> evaluate, then for
/// `num_gens` generations: rank + crowd -> sort by (rank, crowding) -> copy
/// top `10%` as elites -> fill the rest via roulette-selected parents ->
/// crossover (prob. `cross_rate`) or clone -> repaired mutation (prob.
/// `mut_rate`, applied regardless of crossover/clone) -> evaluate -> replace
/// the whole population (size stays exactly `pop_size`, never a 2N pool).
/// Returns the final rank-assigned population; the caller applies the
/// paper's max-modularity rank-1 decision rule.
pub fn run(
    graph: &Graph,
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
        .map(|_| make_individual(graph, locus, locus.random_genome(&mut rng), r, alpha))
        .collect();

    // "elite reproduction 10% of the population size".
    let elite_count = ((0.10 * pop_size as f64).round() as usize).max(1).min(pop_size);

    for _gen in 0..num_gens {
        fast_non_dominated_sort(&mut pop);
        calculate_crowding_distance(&mut pop);

        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            pop[a].rank.cmp(&pop[b].rank).then_with(|| {
                pop[b]
                    .crowding_distance
                    .partial_cmp(&pop[a].crowding_distance)
                    .unwrap_or(Ordering::Equal)
            })
        });

        // Rank-based scalar fitness over the 1-based position in `order`
        // (MATLAB gamultiobj-style rank fitness scaling; the paper itself
        // only says "roulette selection function" without giving the exact
        // internal formula).
        let fitness: Vec<f64> = (0..order.len()).map(|pos| 1.0 / ((pos + 1) as f64).sqrt()).collect();
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
            next_gen.push(make_individual(graph, locus, child_genome, r, alpha));
        }

        pop = next_gen;
    }

    fast_non_dominated_sort(&mut pop);
    pop
}
