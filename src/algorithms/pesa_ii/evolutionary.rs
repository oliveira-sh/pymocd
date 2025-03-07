//! algorithms/pesa_ii/evolutionary.rs
//! Implements the first phase of the algorithm (Genetic algorithm)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::algorithms::pesa_ii::{hypergrid, HyperBox, Solution};
use crate::operators::*;

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

pub const MAX_ARCHIVE_SIZE: usize = 100;

/// Thread-safe random number generator management
struct SafeRng {
    seed_counter: AtomicU64,
}

impl SafeRng {
    fn new() -> Self {
        Self {
            seed_counter: AtomicU64::new(0),
        }
    }

    fn get_rng(&self) -> ChaCha8Rng {
        let seed = self.seed_counter.fetch_add(1, Ordering::SeqCst);
        ChaCha8Rng::seed_from_u64(seed)
    }
}

/// Parallel population generation using PESA-II selection and reproduction
fn generate_new_population(
    hyperboxes: &[HyperBox],
    args: &AGArgs,
    graph: &Graph,
) -> Vec<Partition> {
    // Create thread-safe RNG
    let safe_rng = Arc::new(SafeRng::new());

    // Calculate chunk size based on available threads
    let chunk_size = (args.pop_size / rayon::current_num_threads()).max(1);

    // Create chunks for parallel processing
    let chunks: Vec<_> = (0..args.pop_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| c.len())
        .collect();

    // Process chunks in parallel
    let results: Vec<Partition> = chunks
        .par_iter()
        .flat_map(|&chunk_size| {
            let mut local_children = Vec::with_capacity(chunk_size);
            let safe_rng = Arc::clone(&safe_rng);

            // Create a thread-local RNG
            let mut local_rng = safe_rng.get_rng();

            // Process each chunk
            for _ in 0..chunk_size {
                // Thread-safe selection of parents
                let parent1 = hypergrid::select(hyperboxes, &mut local_rng);
                let parent2 = hypergrid::select(hyperboxes, &mut local_rng);

                // Perform crossover and mutation
                let mut child = crossover(&parent1.partition, &parent2.partition, args.cross_rate);
                mutation(&mut child, graph, args.mut_rate);
                local_children.push(child);
            }

            local_children
        })
        .collect();

    results
}

pub fn evolutionary_phase(
    graph: &Graph,
    args: &AGArgs,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (Vec<Solution>, Vec<f64>) {
    // Validate graph
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        println!("[evolutionary_phase]: Empty graph detected");
        return (Vec::new(), Vec::new());
    }

    // Debug print graph information
    if args.debug >= 2 {
        println!(
            "[evolutionary_phase]: Starting with graph - nodes: {}, edges: {}",
            graph.nodes.len(),
            graph.edges.len()
        );
    }

    let mut archive: Vec<Solution> = Vec::with_capacity(args.pop_size);
    
    // Generate and validate initial population
    let mut population = generate_population(graph, args.pop_size);
    if population.is_empty() {
        println!("[evolutionary_phase]: Failed to generate initial population");
        return (Vec::new(), Vec::new());
    }

    if args.debug >= 2 {
        println!(
            "[evolutionary_phase]: Initial population size: {}",
            population.len()
        );
    }

    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);
    let mut max_local: ConvergenceCriteria = ConvergenceCriteria::default();

    for generation in 0..args.num_gens {
        // Validate population size before parallel processing
        let num_threads = rayon::current_num_threads();
        let chunk_size = population.len().max(1) / num_threads;
        
        if chunk_size == 0 {
            println!("[evolutionary_phase]: Population too small for parallelization");
            break;
        }

        // Evaluate current population and update archive
        let solutions: Vec<Solution> = population
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|partition| {
                        let metrics = get_fitness(graph, partition, degrees, true);
                        Solution {
                            partition: partition.clone(),
                            objectives: vec![metrics.inter, metrics.intra],
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        if solutions.is_empty() {
            println!("[evolutionary_phase]: No valid solutions generated");
            break;
        }

        // Update Pareto archive
        for solution in solutions {
            if !archive.iter().any(|archived| archived.dominates(&solution)) {
                archive.retain(|archived| !solution.dominates(archived));
                archive.push(solution);
            }
        }

        if archive.is_empty() {
            println!("[evolutionary_phase]: Empty archive after update");
            break;
        }

        if archive.len() > MAX_ARCHIVE_SIZE {
            hypergrid::truncate_archive(&mut archive, MAX_ARCHIVE_SIZE);
        }

        // Validate archive before creating hyperboxes
        if archive.is_empty() {
            println!("[evolutionary_phase]: Empty archive after truncation");
            break;
        }

        // Create hyperboxes from archive
        let hyperboxes: Vec<HyperBox> = hypergrid::create(&archive, hypergrid::GRID_DIVISIONS);
        
        if hyperboxes.is_empty() {
            println!("[evolutionary_phase]: No valid hyperboxes created");
            break;
        }

        // Safely compute best fitness
        let best_fitness = archive
            .iter()
            .map(|s| 1.0 - s.objectives[0] - s.objectives[1])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::NEG_INFINITY);

        best_fitness_history.push(best_fitness);

        // Generate new population with validation
        let new_population = generate_new_population(&hyperboxes, args, graph);
        if new_population.is_empty() {
            println!("[evolutionary_phase]: Failed to generate new population");
            break;
        }
        population = new_population;

        // Early stopping
        if max_local.has_converged(best_fitness) {
            if args.debug >= 1{
                println!("[evolutionary_phase]: Converged!");
            }
            break;
        }

        if args.debug >= 1{
            println!(
                "\x1b[1A\x1b[2K[evolutionary_phase]: gen: {} | bf: {:.4} | pop/arch: {}/{} | bA: {:.4} |",
                generation,
                best_fitness,
                population.len(),
                archive.len(),
                max_local.get_best_fitness(),
            );
        }
    }

    // Return empty results if archive is empty
    if archive.is_empty() {
        return (Vec::new(), best_fitness_history);
    }

    (archive, best_fitness_history)
}