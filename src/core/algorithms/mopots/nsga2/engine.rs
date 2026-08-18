//! The mopots generational loop: initialise, then breed and select for `num_gens`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mopots::operators::generate_population;
use crate::core::graph::Graph;

use super::individual::Individual;
use super::offspring::create_offspring;
use super::sorting::fast_non_dominated_sort;
use super::survival::select_survivors;

/// NSGA-II generational loop. `evaluate` sets each individual's `objectives` to
/// `[cut, pair]`. `on_generation(gen, num_gens, &pop)` runs after each generation's
/// offspring are merged. Returns the final combined population **unfiltered** — the
/// caller applies its own rank-1 filter and selection.
#[allow(clippy::too_many_arguments)]
pub fn evolve(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    tournament_size: usize,
    mut evaluate: impl FnMut(&mut [Individual]),
    mut on_generation: impl FnMut(usize, usize, &[Individual]),
) -> Vec<Individual> {
    let mut individuals: Vec<Individual> = generate_population(graph, pop_size)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    evaluate(&mut individuals);

    for generation in 0..num_gens {
        select_survivors(&mut individuals, pop_size);

        let mut offspring = create_offspring(
            &individuals,
            graph,
            cross_rate,
            mut_rate,
            tournament_size,
            generation,
        );
        evaluate(&mut offspring);

        individuals.extend(offspring);

        on_generation(generation, num_gens, &individuals);
    }

    // the loop only ranks at the top of a generation, so rank the pool the caller filters
    fast_non_dominated_sort(&mut individuals);

    individuals
}
