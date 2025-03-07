//! algorithms/nsga_ii/main.rs
//! Implements the Pareto Envelope-based Selection Algorithm II (PESA-II)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::algorithms::nsga_ii::individual::*;
use crate::algorithms::nsga_ii::core::*;
use crate::utils::args::AGArgs;
use crate::graph::{Graph, Partition};
use crate::operators;
use rayon::prelude::*;
use std::time::Instant;
use std::cmp::Ordering;

const TOURNAMENT_SIZE: usize = 2;

// NSGA-II with max-Q from Pareto front
pub fn nsga_ii(graph: &Graph, debug_level: i8, args: AGArgs) -> Partition {
    let start_time = Instant::now();
    
    // Precompute node degrees once
    let degrees = graph.precompute_degress();
    
    // Generate initial population
    let mut individuals: Vec<Individual> = operators::generate_population(graph, args.pop_size)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    
    // Evaluate initial population in parallel
    individuals.par_iter_mut().for_each(|ind| {
        let metrics = operators::get_fitness(graph, &ind.partition, &degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
        ind.calculate_fitness();
    });
    
    let mut best_fitness_history = Vec::with_capacity(args.num_gens);
    
    // Main generational loop
    let mut max_local = operators::ConvergenceCriteria::default();
    for generation in 0..args.num_gens {
        // Sort by rank and crowding distance
        fast_non_dominated_sort(&mut individuals);
        calculate_crowding_distance(&mut individuals);
        
        // Create offspring population
        let mut offspring = create_offspring(
            &individuals, 
            graph, 
            args.cross_rate,
            args.mut_rate,
            TOURNAMENT_SIZE
        );
        
        // Evaluate offspring in parallel
        offspring.par_iter_mut().for_each(|ind| {
            let metrics = operators::get_fitness(graph, &ind.partition, &degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
            ind.calculate_fitness();
        });
        
        // Reserve capacity before extending
        let combined_size = individuals.len() + offspring.len();
        if individuals.capacity() < combined_size {
            individuals.reserve(combined_size - individuals.capacity());
        }
        
        // Combine populations
        individuals.extend(offspring);
        
        // Apply selection (environmental selection)
        fast_non_dominated_sort(&mut individuals);
        calculate_crowding_distance(&mut individuals);
        
        // Sort by rank and crowding distance with unstable sort for speed
        individuals.sort_unstable_by(|a, b| {
            a.rank.cmp(&b.rank).then_with(|| {
                b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(Ordering::Equal)
            })
        });
        
        // Reduce to population size
        individuals.truncate(args.pop_size);
        
        // Track best fitness more efficiently
        let best_fitness = individuals.par_iter()
            .map(|ind| ind.fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f64::NEG_INFINITY);
        
        best_fitness_history.push(best_fitness);


        // Early stopping
        if max_local.has_converged(best_fitness) {
            if args.debug >= 1{
                println!("[evolutionary_phase]: Converged!");
            }
                break;
        }
        
        if debug_level >= 1 && (generation % 10 == 0 || generation == args.num_gens - 1) {
            let first_front_size = individuals.iter().filter(|ind| ind.rank == 1).count();
            println!(
                "NSGA-II: Gen {} | Best fitness: {:.4} | First front size: {} | Pop size: {}",
                generation,
                best_fitness,
                first_front_size,
                individuals.len()
            );
        }
    }
    
    // Extract Pareto front (first non-dominated front)
    let first_front: Vec<Individual> = individuals.iter()
        .filter(|ind| ind.rank == 1)
        .cloned()
        .collect();
    
    // Select solution with maximum Q value from Pareto front
    let best_solution = max_q_selection(&first_front);
    
    let elapsed = start_time.elapsed();
    if debug_level >= 1 {
        println!("NSGA-II completed in {:.2?}", elapsed);
    }
    
    best_solution.partition.clone()
}