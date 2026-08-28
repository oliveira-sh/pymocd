//! The generational loop and the paper's two search customizations.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt, rng};
use std::collections::HashSet;

use crate::core::algorithms::ccm::locus::{self, canonical_labels};

use super::individual::{Individual, fast_non_dominated_sort, new_individual};
use super::reference_points::das_dennis;
use super::survival::environmental_selection;

/// `evaluate` maps decoded labels to their objective vector. Returns the final,
/// rank-sorted population.
pub fn evolve<F>(
    neighbor_pos: &[Vec<usize>],
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
    mut evaluate: F,
) -> Vec<Individual>
where
    F: FnMut(&[i32]) -> Vec<f64>,
{
    let mut r = rng();

    let mut pop: Vec<Individual> = (0..pop_size)
        .map(|_| new_individual(locus::random_genome(neighbor_pos, &mut r)))
        .collect();
    for ind in &mut pop {
        ind.objectives = evaluate(&ind.labels);
    }
    fast_non_dominated_sort(&mut pop);

    let m = pop.first().map_or(0, |i| i.objectives.len());
    let ref_points = das_dennis(m, divisions);

    for _ in 0..num_gens {
        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            offspring.push(breed(&pop, neighbor_pos, cross_rate, mut_rate, &mut r));
        }
        for ind in &mut offspring {
            ind.objectives = evaluate(&ind.labels);
        }
        let mut combined = pop;
        combined.extend(offspring);

        pop = environmental_selection(combined, pop_size, &ref_points, &mut r);

        apply_customizations(&mut pop, neighbor_pos, &mut r, &mut evaluate);
        // re-rank: the customizations replaced members after selection ranked them
        fast_non_dominated_sort(&mut pop);
    }

    pop
}

fn breed(
    pop: &[Individual],
    neighbor_pos: &[Vec<usize>],
    cross_rate: f64,
    mut_rate: f64,
    rng: &mut impl Rng,
) -> Individual {
    // parents are drawn uniformly, with no mating selection: that is pymoo's
    // NSGA-III for unconstrained problems, which the paper adapted
    let pa = rng.random_range(0..pop.len());
    let pb = rng.random_range(0..pop.len());
    let mut child_genome = if rng.random_bool(cross_rate) {
        locus::uniform_crossover(&pop[pa].genome, &pop[pb].genome, rng)
    } else {
        let clone_from = if rng.random_bool(0.5) { pa } else { pb };
        pop[clone_from].genome.clone()
    };
    locus::mutate(&mut child_genome, neighbor_pos, mut_rate, rng);
    new_individual(child_genome)
}

/// Shaik, Ravi & Deb 2021, Sec. 4, in the order the paper applies them.
fn apply_customizations<F>(
    pop: &mut [Individual],
    neighbor_pos: &[Vec<usize>],
    rng: &mut impl Rng,
    evaluate: &mut F,
) where
    F: FnMut(&[i32]) -> Vec<f64>,
{
    replace_duplicate_permutations(pop, neighbor_pos, rng, evaluate);
    replace_single_community(pop, neighbor_pos, rng, evaluate);
}

fn replace_duplicate_permutations<F>(
    pop: &mut [Individual],
    neighbor_pos: &[Vec<usize>],
    rng: &mut impl Rng,
    evaluate: &mut F,
) where
    F: FnMut(&[i32]) -> Vec<f64>,
{
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    for ind in pop.iter_mut() {
        if !seen.insert(canonical_labels(&ind.labels)) {
            replace_with_random(ind, neighbor_pos, rng, evaluate);
        }
    }
}

fn replace_single_community<F>(
    pop: &mut [Individual],
    neighbor_pos: &[Vec<usize>],
    rng: &mut impl Rng,
    evaluate: &mut F,
) where
    F: FnMut(&[i32]) -> Vec<f64>,
{
    for ind in pop.iter_mut() {
        let single = ind
            .labels
            .first()
            .is_some_and(|&first| ind.labels.iter().all(|&c| c == first));
        if single {
            replace_with_random(ind, neighbor_pos, rng, evaluate);
        }
    }
}

fn replace_with_random<F>(
    ind: &mut Individual,
    neighbor_pos: &[Vec<usize>],
    rng: &mut impl Rng,
    evaluate: &mut F,
) where
    F: FnMut(&[i32]) -> Vec<f64>,
{
    *ind = new_individual(locus::random_genome(neighbor_pos, rng));
    ind.objectives = evaluate(&ind.labels);
}
