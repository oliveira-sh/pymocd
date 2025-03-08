use crate::graph::{Graph, Partition};

use crate::operators;

use rand::{prelude::*, rng};
use rayon::prelude::*;
use rustc_hash::FxHashSet as HashSet;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

const ENSEMBLE_SIZE: usize = 4;

#[derive(Clone, Debug)]
pub struct Individual {
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
    pub fitness: f64,
}

impl Individual {
    pub fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
            fitness: f64::NEG_INFINITY,
        }
    }

    // Check if this individual dominates another
    #[inline(always)]
    pub fn dominates(&self, other: &Individual) -> bool {
        let mut at_least_one_better = false;

        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    #[inline(always)]
    pub fn calculate_fitness(&mut self) {
        self.fitness = 1.0 - self.objectives[0] - self.objectives[1];
    }
}

// Tournament selection with early return
#[inline]
pub fn tournament_selection<'a>(
    population: &'a [Individual],
    tournament_size: usize,
) -> &'a Individual {
    let mut rng: ThreadRng = rng();
    let best_idx: usize = rng.random_range(0..population.len());
    let mut best: &Individual = &population[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx: usize = rng.random_range(0..population.len());
        let candidate: &Individual = &population[candidate_idx];

        if candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance)
        {
            best = candidate;
        }
    }

    best
}

// Create offspring with better parallelization
pub fn create_offspring(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
) -> Vec<Individual> {
    let pop_size = population.len();
    let mut offspring = Vec::with_capacity(pop_size);
    let num_threads = rayon::current_num_threads();
    let chunk_size = (pop_size + num_threads - 1) / num_threads;

    // Use atomic counter for better load balancing
    let offspring_counter = AtomicUsize::new(0);

    let thread_offsprings: Vec<Vec<Individual>> = (0..num_threads)
        .into_par_iter()
        .map(|_| {
            let mut local_rng = rng();
            let mut local_offspring = Vec::with_capacity(chunk_size);

            while offspring_counter.fetch_add(1, AtomicOrdering::Relaxed) < pop_size {
                // Select unique parents
                let mut parents = Vec::with_capacity(ENSEMBLE_SIZE);
                let mut selected_ids =
                    HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, Default::default());

                let mut attempts = 0;
                while parents.len() < ENSEMBLE_SIZE && attempts < 50 {
                    let parent = tournament_selection(population, tournament_size);
                    if selected_ids.insert(parent.rank) {
                        parents.push(parent);
                    }
                    attempts += 1;
                }

                // Fill remaining slots if needed
                while parents.len() < ENSEMBLE_SIZE {
                    parents.push(tournament_selection(population, tournament_size));
                }

                let parent_partitions: Vec<Partition> =
                    parents.iter().map(|p| p.partition.clone()).collect();

                let parent_slice: &[Partition] = &parent_partitions;
                let should_crossover = local_rng.random::<f64>() < crossover_rate;

                let mut child = if should_crossover {
                    operators::ensemble_crossover(parent_slice, 1.0)
                } else {
                    parent_partitions[0].clone()
                };

                operators::mutation(&mut child, graph, mutation_rate);
                local_offspring.push(Individual::new(child));
            }

            local_offspring
        })
        .collect();

    // Combine results, only taking what we need
    let mut remaining = pop_size;
    for mut thread_offspring in thread_offsprings {
        let to_take = remaining.min(thread_offspring.len());
        offspring.extend(thread_offspring.drain(..to_take));
        remaining -= to_take;
        if remaining == 0 {
            break;
        }
    }

    offspring
}
