//! NSGA-II survivor selection + generational loop (Deb et al. 2002) over the
//! shared `helpers::individual` core; only crowding distance and the
//! crowding-based truncation are NSGA-II-specific and live here.
//! SCALE keeps its own dense-CSR NSGA-II — that representation is the source
//! of its speed and is intentionally NOT shared here.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;
use crate::core::metaheuristics::helpers::individual::{
    Individual, create_offspring, fast_non_dominated_sort,
};
use crate::core::metaheuristics::helpers::operators;
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

pub fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let n_obj = population[0].objectives.len();

    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for (idx, ind) in population.iter().enumerate() {
        rank_groups.entry(ind.rank).or_default().push(idx);
    }

    for indices in rank_groups.values() {
        if indices.len() <= 2 {
            for &i in indices {
                population[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }

        for obj_idx in 0..n_obj {
            let mut sorted = indices.clone();
            sorted.sort_unstable_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            population[sorted[0]].crowding_distance = f64::INFINITY;
            population[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;

            let obj_min = population[sorted[0]].objectives[obj_idx];
            let obj_max = population[sorted[sorted.len() - 1]].objectives[obj_idx];

            if (obj_max - obj_min).abs() > f64::EPSILON {
                let scale = 1.0 / (obj_max - obj_min);
                for i in 1..sorted.len() - 1 {
                    let prev_obj = population[sorted[i - 1]].objectives[obj_idx];
                    let next_obj = population[sorted[i + 1]].objectives[obj_idx];
                    population[sorted[i]].crowding_distance += (next_obj - prev_obj) * scale;
                }
            }
        }
    }
}

/// NSGA-II survivor selection: non-dominated sort + crowding, then keep the best
/// `pop_size` by (rank ascending, crowding descending).
pub fn select_survivors(population: &mut Vec<Individual>, pop_size: usize) {
    fast_non_dominated_sort(population);
    calculate_crowding_distance(population);
    population.sort_unstable_by(|a, b| {
        a.rank.cmp(&b.rank).then_with(|| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });
    population.truncate(pop_size);
}

/// Generic NSGA-II generational loop (Deb et al. 2002). `evaluate` sets each
/// individual's `objectives`. `on_generation(gen, num_gens, &pop)` runs after each
/// generation's offspring are merged. Returns the final combined population
/// **unfiltered** — the caller applies its own rank-1 filter and selection.
/// Generic over error type `E` so pyo3 callers can propagate `PyErr`.
#[allow(clippy::too_many_arguments)]
pub fn evolve<E>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    tournament_size: usize,
    mut evaluate: impl FnMut(&mut [Individual]) -> Result<(), E>,
    mut on_generation: impl FnMut(usize, usize, &[Individual]) -> Result<(), E>,
) -> Result<Vec<Individual>, E> {
    use rayon::prelude::*;

    let mut individuals: Vec<Individual> = operators::generate_population(graph, pop_size)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    evaluate(&mut individuals)?;

    for generation in 0..num_gens {
        select_survivors(&mut individuals, pop_size);

        let mut offspring =
            create_offspring(&individuals, graph, cross_rate, mut_rate, tournament_size);
        evaluate(&mut offspring)?;

        individuals.extend(offspring);

        on_generation(generation, num_gens, &individuals)?;
    }

    Ok(individuals)
}
