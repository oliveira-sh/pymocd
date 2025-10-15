//! Selection utilities for HP-MOCD algorithm
//!
//! This module provides the selection and ranking functions used in the HP-MOCD algorithm,
//! including NSGA-II's fast non-dominated sorting and crowding distance calculation.

use super::individual::Individual;
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

/// Performs fast non-dominated sorting on the population
///
/// This implements the NSGA-II fast non-dominated sorting algorithm,
/// which assigns a rank to each individual based on their Pareto dominance level.
///
/// # Arguments
/// * `population` - Population to sort (modified in-place)
pub fn fast_non_dominated_sort(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }
    fast_non_dominated_sort_nd(population);
}

/// Internal implementation of fast non-dominated sorting with parallel processing
fn fast_non_dominated_sort_nd(population: &mut [Individual]) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = population.len();
    let mut fronts: Vec<Vec<usize>> = Vec::with_capacity(n / 2);
    fronts.push(Vec::with_capacity(n / 2));

    let mut dominated_data = Vec::new();
    let mut dominated_ranges = Vec::with_capacity(n);
    let domination_count: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();

    // Calculate domination relationships in parallel
    let domination_relations: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::new();
            let mut count = 0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                if population[i].dominates(&population[j]) {
                    dominated.push(j);
                } else if population[j].dominates(&population[i]) {
                    count += 1;
                }
            }

            (dominated, count)
        })
        .collect();

    // Build the first front and domination data structures
    for (i, (dominated, count)) in domination_relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        dominated_ranges.push(start..dominated_data.len());
        domination_count[i].store(count, Ordering::Relaxed);

        if count == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }

    // Build subsequent fronts
    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = &fronts[front_idx];
        let next_front: Vec<usize> = current_front
            .par_iter()
            .fold(Vec::new, |mut acc, &i| {
                let range = &dominated_ranges[i];
                for &j in &dominated_data[range.start..range.end] {
                    let prev = domination_count[j].fetch_sub(1, Ordering::Relaxed);
                    if prev == 1 {
                        acc.push(j);
                    }
                }
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        front_idx += 1;
        if !next_front.is_empty() {
            for &j in &next_front {
                population[j].rank = front_idx + 1;
            }
            fronts.push(next_front);
        } else {
            break;
        }
    }
}

/// Calculates crowding distance for diversity preservation
///
/// Crowding distance measures how close an individual is to its neighbors
/// in the objective space. Higher values indicate more isolated (diverse) solutions.
///
/// # Arguments
/// * `population` - Population to calculate distances for (modified in-place)
pub fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let n_obj = population[0].objectives.len();

    // Initialize all crowding distances to 0
    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    // Group individuals by rank
    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for (idx, ind) in population.iter().enumerate() {
        rank_groups.entry(ind.rank).or_default().push(idx);
    }

    // Calculate crowding distance for each rank group separately
    for indices in rank_groups.values() {
        if indices.len() <= 2 {
            // If there are only 1-2 individuals in the front, set infinite distance
            for &i in indices {
                population[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }

        // Calculate distance for each objective
        for obj_idx in 0..n_obj {
            let mut sorted = indices.clone();
            sorted.sort_unstable_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinite distance
            population[sorted[0]].crowding_distance = f64::INFINITY;
            population[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;

            let obj_min = population[sorted[0]].objectives[obj_idx];
            let obj_max = population[sorted[sorted.len() - 1]].objectives[obj_idx];

            // Calculate distances for intermediate points
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

/// Calculates the Q-value (modularity) for an individual
///
/// Q-value is calculated as 1 - intra - inter, where intra and inter
/// are the normalized objective values.
///
/// # Arguments
/// * `ind` - Individual to calculate Q-value for
///
/// # Returns
/// Q-value (modularity) of the individual
#[inline(always)]
pub fn q(ind: &Individual) -> f64 {
    1.0 - ind.objectives[0] - ind.objectives[1]
}

/// Selects the individual with the highest Q-value (modularity)
///
/// This function implements the final selection strategy for HP-MOCD,
/// choosing the solution with the best modularity from the Pareto front.
///
/// # Arguments
/// * `population` - Population to select from
///
/// # Returns
/// Reference to the individual with highest Q-value
///
/// # Panics
/// Panics if the population is empty
#[inline(always)]
pub fn max_q_selection(population: &[Individual]) -> &Individual {
    population
        .iter()
        .max_by(|a, b| q(a).partial_cmp(&q(b)).unwrap_or(Ordering::Equal))
        .expect("Empty population")
}

/// Tournament selection for parent selection
///
/// Selects the best individual from a randomly chosen subset of the population.
///
/// # Arguments
/// * `population` - Population to select from
/// * `tournament_size` - Number of individuals to compare
///
/// # Returns
/// Index of the selected individual
pub fn tournament_selection(population: &[Individual], tournament_size: usize) -> usize {
    use rand::prelude::*;

    let mut rng = rand::rng();
    let mut best_idx = rng.random_range(0..population.len());
    let mut best = &population[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx = rng.random_range(0..population.len());
        let candidate = &population[candidate_idx];

        if candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance)
        {
            best = candidate;
            best_idx = candidate_idx;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    fn create_test_individual(objectives: [f64; 2]) -> Individual {
        let mut partition = FxHashMap::default();
        partition.insert(0, 0);
        Individual {
            partition,
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    #[test]
    fn test_q_value_calculation() {
        let ind = create_test_individual([0.2, 0.3]);
        assert_eq!(q(&ind), 0.5); // 1.0 - 0.2 - 0.3
    }

    #[test]
    fn test_max_q_selection() {
        let population = vec![
            create_test_individual([0.3, 0.4]), // Q = 0.3
            create_test_individual([0.1, 0.2]), // Q = 0.7
            create_test_individual([0.2, 0.3]), // Q = 0.5
        ];

        let best = max_q_selection(&population);
        assert_eq!(best.objectives, [0.1, 0.2]);
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let mut population = vec![
            create_test_individual([1.0, 3.0]), // Front 1
            create_test_individual([2.0, 2.0]), // Front 1
            create_test_individual([3.0, 1.0]), // Front 1
            create_test_individual([2.0, 3.0]), // Front 2 (dominated by first)
            create_test_individual([4.0, 4.0]), // Front 3
        ];

        fast_non_dominated_sort(&mut population);

        // Check first front
        assert_eq!(population[0].rank, 1);
        assert_eq!(population[1].rank, 1);
        assert_eq!(population[2].rank, 1);

        // Check later fronts
        assert!(population[3].rank > 1);
        assert!(population[4].rank > 1);
    }

    #[test]
    fn test_crowding_distance() {
        let mut population = vec![
            create_test_individual([1.0, 5.0]), // Boundary
            create_test_individual([2.0, 4.0]), // Middle
            create_test_individual([3.0, 3.0]), // Middle
            create_test_individual([4.0, 2.0]), // Middle
            create_test_individual([5.0, 1.0]), // Boundary
        ];

        // Set all to same rank for distance calculation
        for ind in &mut population {
            ind.rank = 1;
        }

        calculate_crowding_distance(&mut population);

        // Boundary points should have infinite distance
        assert_eq!(population[0].crowding_distance, f64::INFINITY);
        assert_eq!(population[4].crowding_distance, f64::INFINITY);

        // Middle points should have finite positive distances
        assert!(population[1].crowding_distance > 0.0);
        assert!(population[2].crowding_distance > 0.0);
        assert!(population[3].crowding_distance > 0.0);
    }

    #[test]
    fn test_tournament_selection() {
        let population = vec![
            create_test_individual([1.0, 1.0]),
            create_test_individual([2.0, 2.0]),
            create_test_individual([3.0, 3.0]),
        ];

        // Set different ranks
        let mut population = population;
        population[0].rank = 1;
        population[1].rank = 2;
        population[2].rank = 3;

        let selected = tournament_selection(&population, 2);
        // Can't guarantee specific result due to randomness, but should be valid index
        assert!(selected < population.len());
    }
}
