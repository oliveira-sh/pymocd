//! Selection operators for evolutionary algorithms
//! 
//! This module implements various selection strategies used in multiobjective
//! evolutionary algorithms, including tournament selection, non-dominated sorting,
//! and crowding distance calculation.

use rand::prelude::*;
use std::cmp::Ordering;

/// Represents an individual solution in the population
/// 
/// This trait should be implemented by solution types that can be used
/// with the selection operators in this module.
pub trait Individual {
    /// Get the objective values for this individual
    fn objectives(&self) -> &[f64];
    
    /// Get the rank (Pareto front number) of this individual
    fn rank(&self) -> usize;
    
    /// Set the rank of this individual
    fn set_rank(&mut self, rank: usize);
    
    /// Get the crowding distance of this individual
    fn crowding_distance(&self) -> f64;
    
    /// Set the crowding distance of this individual
    fn set_crowding_distance(&mut self, distance: f64);
}

/// Performs tournament selection among individuals
/// 
/// Selects the best individual from a randomly chosen subset of the population
/// based on Pareto dominance and crowding distance.
/// 
/// # Arguments
/// * `population` - The population to select from
/// * `tournament_size` - Number of individuals in each tournament
/// 
/// # Returns
/// Index of the selected individual
pub fn tournament_selection<T: Individual>(population: &[T], tournament_size: usize) -> usize {
    let mut rng = rand::rng();
    let mut best_idx = rng.random_range(0..population.len());
    
    for _ in 1..tournament_size {
        let candidate_idx = rng.random_range(0..population.len());
        
        if is_better(&population[candidate_idx], &population[best_idx]) {
            best_idx = candidate_idx;
        }
    }
    
    best_idx
}

/// Determines if individual A is better than individual B
/// 
/// Uses Pareto dominance first, then crowding distance for tie-breaking.
fn is_better<T: Individual>(a: &T, b: &T) -> bool {
    match a.rank().cmp(&b.rank()) {
        Ordering::Less => true,  // Lower rank is better (closer to Pareto front)
        Ordering::Greater => false,
        Ordering::Equal => {
            // Same rank, use crowding distance (higher is better for diversity)
            a.crowding_distance() > b.crowding_distance()
        }
    }
}

/// Performs fast non-dominated sorting (NSGA-II algorithm)
/// 
/// Assigns rank (Pareto front number) to each individual in the population.
/// Rank 1 indicates the first (best) Pareto front.
/// 
/// # Arguments
/// * `population` - The population to sort (modified in-place)
/// 
/// # Type Parameters
/// * `T` - Individual type that implements the Individual trait
pub fn fast_non_dominated_sort<T: Individual>(population: &mut [T]) {
    let n = population.len();
    
    // Domination data structures
    let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_counts = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front = Vec::new();
    
    // Calculate domination relationships
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            
            if dominates(population[i].objectives(), population[j].objectives()) {
                dominated_solutions[i].push(j);
            } else if dominates(population[j].objectives(), population[i].objectives()) {
                domination_counts[i] += 1;
            }
        }
        
        if domination_counts[i] == 0 {
            population[i].set_rank(1);
            first_front.push(i);
        }
    }
    
    fronts.push(first_front);
    let mut current_front = 0;
    
    // Build subsequent fronts
    while !fronts[current_front].is_empty() {
        let mut next_front = Vec::new();
        
        for &individual in &fronts[current_front] {
            for &dominated in &dominated_solutions[individual] {
                domination_counts[dominated] -= 1;
                
                if domination_counts[dominated] == 0 {
                    population[dominated].set_rank(current_front + 2);
                    next_front.push(dominated);
                }
            }
        }
        
        current_front += 1;
        fronts.push(next_front);
    }
}

/// Checks if objective vector `a` dominates objective vector `b`
/// 
/// For minimization problems: A dominates B if A is better or equal in all objectives
/// and strictly better in at least one objective.
fn dominates(a: &[f64], b: &[f64]) -> bool {
    assert_eq!(a.len(), b.len());
    
    let mut at_least_one_better = false;
    
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        if a_val > b_val {
            // A is worse in this objective (assuming minimization)
            return false;
        } else if a_val < b_val {
            at_least_one_better = true;
        }
    }
    
    at_least_one_better
}

/// Calculates crowding distance for individuals in the same Pareto front
/// 
/// Crowding distance measures how close an individual is to its neighbors.
/// Higher values indicate more isolated (diverse) solutions.
/// 
/// # Arguments
/// * `population` - The population to calculate distances for (modified in-place)
/// 
/// # Type Parameters
/// * `T` - Individual type that implements the Individual trait
pub fn calculate_crowding_distance<T: Individual>(population: &mut [T]) {
    let n = population.len();
    
    if n == 0 {
        return;
    }
    
    // Initialize all distances to 0
    for individual in population.iter_mut() {
        individual.set_crowding_distance(0.0);
    }
    
    // Get number of objectives
    let num_objectives = population[0].objectives().len();
    
    // Calculate distance for each objective
    for obj_idx in 0..num_objectives {
        // Sort by this objective
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            population[a].objectives()[obj_idx]
                .partial_cmp(&population[b].objectives()[obj_idx])
                .unwrap_or(Ordering::Equal)
        });
        
        // Set boundary points to infinite distance
        if n > 2 {
            let first_idx = indices[0];
            let last_idx = indices[n - 1];
            
            population[first_idx].set_crowding_distance(f64::INFINITY);
            population[last_idx].set_crowding_distance(f64::INFINITY);
        }
        
        // Calculate objective range
        let obj_min = population[indices[0]].objectives()[obj_idx];
        let obj_max = population[indices[n - 1]].objectives()[obj_idx];
        let obj_range = obj_max - obj_min;
        
        if obj_range > 0.0 {
            // Calculate distances for intermediate points
            for i in 1..n - 1 {
                let current_idx = indices[i];
                let prev_obj = population[indices[i - 1]].objectives()[obj_idx];
                let next_obj = population[indices[i + 1]].objectives()[obj_idx];
                
                let distance_contribution = (next_obj - prev_obj) / obj_range;
                let current_distance = population[current_idx].crowding_distance();
                population[current_idx].set_crowding_distance(current_distance + distance_contribution);
            }
        }
    }
}

/// Environmental selection based on NSGA-II principles
/// 
/// Selects the best individuals for the next generation using non-dominated
/// sorting and crowding distance.
/// 
/// # Arguments
/// * `population` - Combined population of parents and offspring
/// * `target_size` - Desired population size
/// 
/// # Returns
/// Indices of selected individuals
pub fn environmental_selection<T: Individual>(
    population: &mut [T], 
    target_size: usize
) -> Vec<usize> {
    if population.len() <= target_size {
        return (0..population.len()).collect();
    }
    
    // Perform non-dominated sorting
    fast_non_dominated_sort(population);
    
    // Calculate crowding distances
    calculate_crowding_distance(population);
    
    // Sort by rank first, then by crowding distance
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| {
        population[a].rank().cmp(&population[b].rank()).then_with(|| {
            population[b].crowding_distance()
                .partial_cmp(&population[a].crowding_distance())
                .unwrap_or(Ordering::Equal)
        })
    });
    
    // Select the best individuals
    indices.into_iter().take(target_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock individual for testing
    #[derive(Debug, Clone)]
    struct TestIndividual {
        objectives: Vec<f64>,
        rank: usize,
        crowding_distance: f64,
    }
    
    impl TestIndividual {
        fn new(objectives: Vec<f64>) -> Self {
            TestIndividual {
                objectives,
                rank: 0,
                crowding_distance: 0.0,
            }
        }
    }
    
    impl Individual for TestIndividual {
        fn objectives(&self) -> &[f64] {
            &self.objectives
        }
        
        fn rank(&self) -> usize {
            self.rank
        }
        
        fn set_rank(&mut self, rank: usize) {
            self.rank = rank;
        }
        
        fn crowding_distance(&self) -> f64 {
            self.crowding_distance
        }
        
        fn set_crowding_distance(&mut self, distance: f64) {
            self.crowding_distance = distance;
        }
    }
    
    #[test]
    fn test_dominates() {
        // A dominates B if A is better or equal in all objectives and strictly better in at least one
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0])); // A better in both
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0])); // A equal in first, better in second
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0])); // Trade-off, no dominance
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0])); // Equal, no dominance
    }
    
    #[test]
    fn test_fast_non_dominated_sort() {
        let mut population = vec![
            TestIndividual::new(vec![1.0, 3.0]), // Front 1
            TestIndividual::new(vec![2.0, 2.0]), // Front 1  
            TestIndividual::new(vec![3.0, 1.0]), // Front 1
            TestIndividual::new(vec![2.0, 3.0]), // Front 2 (dominated by first)
            TestIndividual::new(vec![4.0, 4.0]), // Front 3 (dominated by multiple)
        ];
        
        fast_non_dominated_sort(&mut population);
        
        // Check that ranks are assigned correctly
        assert_eq!(population[0].rank(), 1); // First front
        assert_eq!(population[1].rank(), 1); // First front
        assert_eq!(population[2].rank(), 1); // First front
        assert!(population[3].rank() > 1);    // Later front
        assert!(population[4].rank() > 1);    // Later front
    }
    
    #[test]
    fn test_crowding_distance() {
        let mut population = vec![
            TestIndividual::new(vec![1.0, 5.0]), // Boundary
            TestIndividual::new(vec![2.0, 4.0]), // Middle
            TestIndividual::new(vec![3.0, 3.0]), // Middle
            TestIndividual::new(vec![4.0, 2.0]), // Middle
            TestIndividual::new(vec![5.0, 1.0]), // Boundary
        ];
        
        calculate_crowding_distance(&mut population);
        
        // Boundary points should have infinite distance
        assert_eq!(population[0].crowding_distance(), f64::INFINITY);
        assert_eq!(population[4].crowding_distance(), f64::INFINITY);
        
        // Middle points should have finite positive distances
        assert!(population[1].crowding_distance() > 0.0);
        assert!(population[2].crowding_distance() > 0.0);
        assert!(population[3].crowding_distance() > 0.0);
    }
    
    #[test]
    fn test_tournament_selection() {
        let mut population = vec![
            TestIndividual::new(vec![1.0, 1.0]),
            TestIndividual::new(vec![2.0, 2.0]),
            TestIndividual::new(vec![3.0, 3.0]),
        ];
        
        // Set up ranks (lower is better)
        population[0].set_rank(1);
        population[1].set_rank(2);
        population[2].set_rank(3);
        
        // Tournament selection should favor individuals with lower rank
        let selected = tournament_selection(&population, 2);
        // Can't guarantee specific result due to randomness, but test that it runs
        assert!(selected < population.len());
    }
}