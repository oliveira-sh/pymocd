//! The generational loop and KRM's two population customizations.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rng;
use rand::rngs::ThreadRng;
use std::collections::HashSet;

use crate::core::algorithms::krm::locus::{Genome, Locus};
use crate::core::algorithms::krm::objectives::{kkm_ratiocut, modularity};
use crate::core::algorithms::krm::operators::{crossover, mutate, random_parent};
use crate::core::graph::Graph;

use super::individual::{Individual, fast_non_dominated_sort};
use super::reference_points::das_dennis;
use super::survival::environmental_selection;

fn make_individual(graph: &Graph, locus: &Locus, genome: Genome) -> Individual {
    let labels = locus.decode(&genome);
    let (kkm, rc) = kkm_ratiocut(locus, &labels);
    let q = modularity(graph, locus, &labels);
    Individual {
        genome,
        labels,
        objectives: vec![kkm, rc, -q],
        rank: usize::MAX,
    }
}

fn replace_duplicate_permutations(
    graph: &Graph,
    locus: &Locus,
    pop: &mut [Individual],
    rng: &mut ThreadRng,
) {
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    for ind in pop.iter_mut() {
        let canon = locus.canonical_labels(&ind.labels);
        if !seen.insert(canon) {
            *ind = make_individual(graph, locus, locus.random_genome(rng));
        }
    }
}

fn replace_single_community(
    graph: &Graph,
    locus: &Locus,
    pop: &mut [Individual],
    rng: &mut ThreadRng,
) {
    for ind in pop.iter_mut() {
        if locus.is_single_community(&ind.labels) {
            *ind = make_individual(graph, locus, locus.random_genome(rng));
        }
    }
}

/// The final, rank-assigned population; the caller picks from rank 1.
pub fn run(
    graph: &Graph,
    locus: &Locus,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Vec<Individual> {
    let mut rng = rng();

    let mut pop: Vec<Individual> = (0..pop_size)
        .map(|_| make_individual(graph, locus, locus.random_genome(&mut rng)))
        .collect();
    fast_non_dominated_sort(&mut pop);

    let m = pop.first().map_or(0, |i| i.objectives.len());
    let ref_points = das_dennis(m, divisions);

    for _generation in 0..num_gens {
        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let pa = random_parent(pop.len(), &mut rng);
            let pb = random_parent(pop.len(), &mut rng);
            let mut child_genome =
                crossover(&pop[pa].genome, &pop[pb].genome, cross_rate, &mut rng);
            mutate(&mut child_genome, locus, mut_rate, &mut rng);
            offspring.push(make_individual(graph, locus, child_genome));
        }

        let mut combined = std::mem::take(&mut pop);
        combined.extend(offspring);

        pop = environmental_selection(combined, pop_size, &ref_points);

        replace_duplicate_permutations(graph, locus, &mut pop, &mut rng);
        replace_single_community(graph, locus, &mut pop, &mut rng);
        // The replacements above inherit the rank of whoever they overwrote,
        // so the population must be re-ranked before it is bred from or returned.
        fast_non_dominated_sort(&mut pop);
    }

    pop
}
