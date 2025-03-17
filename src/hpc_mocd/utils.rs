use crate::hpc_mocd::individual::Individual;

use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

// Fast non-dominated sort with optimized data structures and parallelism
pub fn fast_non_dominated_sort(population: &mut [Individual]) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let pop_size = population.len();

    // Preallocate fronts
    let mut fronts: Vec<Vec<usize>> = Vec::with_capacity(pop_size / 2);
    fronts.push(Vec::with_capacity(pop_size / 2));

    // Store dominated indices in a contiguous buffer with ranges
    let mut dominated_data = Vec::new();
    let mut dominated_indices = Vec::with_capacity(pop_size);

    // Use atomic counters for parallel front processing
    let domination_count: Vec<AtomicUsize> = (0..pop_size).map(|_| AtomicUsize::new(0)).collect();

    // Parallel computation of domination relationships
    let domination_relations: Vec<_> = (0..pop_size)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::with_capacity(20); // Increased initial capacity
            let mut count = 0;

            for j in 0..pop_size {
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

    // Build contiguous dominated data and indices
    for (i, (dominated, count)) in domination_relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        dominated_indices.push(start..dominated_data.len());
        domination_count[i].store(count, Ordering::Relaxed);

        if count == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }

    // Process fronts in parallel using atomic operations
    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = &fronts[front_idx];
        let next_front: Vec<usize> = current_front
            .par_iter()
            .fold(Vec::new, |mut acc, &i| {
                let range = &dominated_indices[i];
                for &j in &dominated_data[range.start..range.end] {
                    // Atomic decrement and check for transition to 0
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
            // Assign ranks and store the next front
            for &j in &next_front {
                population[j].rank = front_idx + 1;
            }
            fronts.push(next_front);
        } else {
            break;
        }
    }
}

// Calculate crowding distance with optimized memory usage
pub fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let n_obj = population[0].objectives.len();

    // Reset crowding distances
    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    // Group individuals by rank - preallocate with reasonable size
    let mut rank_groups: HashMap<usize, Vec<usize>> =
        HashMap::with_capacity_and_hasher(10, Default::default());
    for (idx, ind) in population.iter().enumerate() {
        rank_groups
            .entry(ind.rank)
            .or_insert_with(|| Vec::with_capacity(population.len() / 4))
            .push(idx);
    }

    // Calculate crowding distance for each rank
    for (_rank, indices) in rank_groups {
        if indices.len() <= 1 {
            for &i in &indices {
                population[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }

        // Process each objective
        for obj_idx in 0..n_obj {
            // Sort indices by objective value
            let mut sorted = indices.clone();
            sorted.sort_unstable_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinity
            population[sorted[0]].crowding_distance = f64::INFINITY;
            population[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;

            // Calculate distance for interior points
            let obj_min = population[sorted[0]].objectives[obj_idx];
            let obj_max = population[sorted[sorted.len() - 1]].objectives[obj_idx];

            if (obj_max - obj_min).abs() > 1e-10 {
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

#[inline]
pub fn max_q_selection(population: &[Individual]) -> &Individual {
    population
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
        .expect("Empty population in max_q_selection")
}
