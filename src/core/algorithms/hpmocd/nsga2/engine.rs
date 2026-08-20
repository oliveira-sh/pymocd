//! The HP-MOCD generational loop: initialise, then select and breed for `num_gens`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::hpmocd::operators::generate_population;
use crate::core::graph::Graph;

use super::individual::Individual;
use super::offspring::create_offspring;
use super::survival::select_survivors;

/// NSGA-II generational loop (Deb et al. 2002). Returns the merged `2 × pop_size`
/// pool with the offspring half still unranked; the caller filters rank 1 itself.
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
    let mut individuals: Vec<Individual> = generate_population(graph, pop_size)
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
