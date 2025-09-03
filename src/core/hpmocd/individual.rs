//! Individual representation for HP-MOCD algorithm
//! 
//! This module defines the Individual struct that represents a solution
//! (community partition) in the evolutionary algorithm.

use crate::graph::{Graph, Partition};
use crate::operators::{crossover, mutation, tournament_selection};

/// Represents an individual solution in the MOCD evolutionary algorithm
/// 
/// Each individual contains a partition (community assignment) and associated
/// fitness metrics used for selection and ranking.
#[derive(Debug, Clone)]
pub struct Individual {
    /// The community partition represented by this individual
    pub partition: Partition,
    /// Objective values [intra, inter] for multiobjective optimization
    pub objectives: [f64; 2],
    /// Pareto front rank (1 = first front, 2 = second front, etc.)
    pub rank: usize,
    /// Crowding distance for diversity preservation
    pub crowding_distance: f64,
}

impl Individual {
    /// Creates a new individual with the given partition
    /// 
    /// # Arguments
    /// * `partition` - The community partition for this individual
    /// 
    /// # Returns
    /// New Individual with initialized values
    pub fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: [0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Creates an individual with specified objective values
    /// 
    /// # Arguments
    /// * `partition` - The community partition
    /// * `objectives` - The objective values [intra, inter]
    pub fn with_objectives(partition: Partition, objectives: [f64; 2]) -> Self {
        Individual {
            partition,
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Returns true if this individual dominates the other
    /// 
    /// For community detection, we minimize both objectives, so A dominates B
    /// if A is better or equal in both objectives and strictly better in at least one.
    pub fn dominates(&self, other: &Individual) -> bool {
        (self.objectives[0] <= other.objectives[0] && self.objectives[1] <= other.objectives[1]) &&
        (self.objectives[0] < other.objectives[0] || self.objectives[1] < other.objectives[1])
    }

    /// Returns the modularity value for this individual
    /// 
    /// Modularity is calculated as 1 - intra - inter for normalized objectives
    pub fn modularity(&self) -> f64 {
        1.0 - self.objectives[0] - self.objectives[1]
    }

    /// Checks if this individual has better rank or crowding distance than another
    /// 
    /// Used for selection: lower rank is better, higher crowding distance is better for same rank.
    pub fn is_better_than(&self, other: &Individual) -> bool {
        if self.rank != other.rank {
            self.rank < other.rank
        } else {
            self.crowding_distance > other.crowding_distance
        }
    }

    /// Returns a copy of the partition
    pub fn get_partition(&self) -> Partition {
        self.partition.clone()
    }

    /// Updates the partition and resets fitness-related fields
    pub fn set_partition(&mut self, partition: Partition) {
        self.partition = partition;
        self.objectives = [0.0, 0.0];
        self.rank = 0;
        self.crowding_distance = 0.0;
    }

    /// Gets the objective values as a slice
    pub fn get_objectives(&self) -> &[f64] {
        &self.objectives
    }

    /// Sets the objective values
    pub fn set_objectives(&mut self, objectives: [f64; 2]) {
        self.objectives = objectives;
    }

    /// Gets the Pareto front rank
    pub fn get_rank(&self) -> usize {
        self.rank
    }

    /// Sets the Pareto front rank
    pub fn set_rank(&mut self, rank: usize) {
        self.rank = rank;
    }

    /// Gets the crowding distance
    pub fn get_crowding_distance(&self) -> f64 {
        self.crowding_distance
    }

    /// Sets the crowding distance
    pub fn set_crowding_distance(&mut self, distance: f64) {
        self.crowding_distance = distance;
    }
}

/// Implement the selection trait for use with generic selection operators
impl crate::operators::selection::Individual for Individual {
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

/// Creates offspring individuals through crossover and mutation
/// 
/// This function implements the reproduction phase of the genetic algorithm,
/// creating new individuals from the current population.
/// 
/// # Arguments
/// * `population` - Current population of individuals
/// * `graph` - Graph structure for mutation operations
/// * `crossover_rate` - Probability of crossover
/// * `mutation_rate` - Probability of mutation
/// * `tournament_size` - Size of tournament for parent selection
/// 
/// # Returns
/// Vector of offspring individuals
pub fn create_offspring(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
) -> Vec<Individual> {
    let population_size = population.len();
    let mut offspring = Vec::with_capacity(population_size);
    let _rng = rand::rng();

    for _ in 0..population_size {
        // Select parents using tournament selection
        let parent1_idx = tournament_selection(population, tournament_size);
        let parent2_idx = tournament_selection(population, tournament_size);

        // Create offspring through crossover
        let mut child_partition = crossover(
            &population[parent1_idx].partition,
            &population[parent2_idx].partition,
            crossover_rate,
        );

        // Apply mutation
        mutation(&mut child_partition, graph, mutation_rate);

        offspring.push(Individual::new(child_partition));
    }

    offspring
}

/// Creates a random individual for initial population
/// 
/// # Arguments
/// * `graph` - Graph to create partition for
/// * `max_communities` - Maximum number of communities to generate
/// 
/// # Returns
/// Individual with random partition
pub fn create_random_individual(graph: &Graph, max_communities: usize) -> Individual {
    let partition = crate::utils::random::random_partition(graph, max_communities);
    Individual::new(partition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rustc_hash::FxHashMap;

    fn create_test_partition() -> Partition {
        let mut partition = FxHashMap::default();
        partition.insert(0, 0);
        partition.insert(1, 0);
        partition.insert(2, 1);
        partition.insert(3, 1);
        partition
    }

    #[test]
    fn test_individual_creation() {
        let partition = create_test_partition();
        let individual = Individual::new(partition.clone());
        
        assert_eq!(individual.partition, partition);
        assert_eq!(individual.objectives, [0.0, 0.0]);
        assert_eq!(individual.rank, 0);
        assert_eq!(individual.crowding_distance, 0.0);
    }

    #[test]
    fn test_individual_with_objectives() {
        let partition = create_test_partition();
        let objectives = [0.3, 0.4];
        let individual = Individual::with_objectives(partition.clone(), objectives);
        
        assert_eq!(individual.partition, partition);
        assert_eq!(individual.objectives, objectives);
        assert!((individual.modularity() - 0.3).abs() < 1e-10); // 1.0 - 0.3 - 0.4
    }

    #[test]
    fn test_dominance() {
        let partition = create_test_partition();
        let ind1 = Individual::with_objectives(partition.clone(), [0.2, 0.3]);
        let ind2 = Individual::with_objectives(partition.clone(), [0.3, 0.4]);
        let ind3 = Individual::with_objectives(partition.clone(), [0.1, 0.5]);
        
        assert!(ind1.dominates(&ind2)); // Better in both objectives
        assert!(!ind2.dominates(&ind1));
        assert!(!ind1.dominates(&ind3)); // Trade-off: better in one, worse in other
        assert!(!ind3.dominates(&ind1));
    }

    #[test]
    fn test_is_better_than() {
        let partition = create_test_partition();
        let mut ind1 = Individual::with_objectives(partition.clone(), [0.2, 0.3]);
        let mut ind2 = Individual::with_objectives(partition.clone(), [0.3, 0.4]);
        
        // Set different ranks
        ind1.set_rank(1);
        ind2.set_rank(2);
        
        assert!(ind1.is_better_than(&ind2)); // Lower rank is better
        
        // Same rank, different crowding distance
        ind2.set_rank(1);
        ind1.set_crowding_distance(0.5);
        ind2.set_crowding_distance(0.3);
        
        assert!(ind1.is_better_than(&ind2)); // Higher crowding distance is better
    }

    #[test]
    fn test_getters_and_setters() {
        let mut individual = Individual::new(create_test_partition());
        
        // Test objective functions
        individual.set_objectives([0.4, 0.5]);
        assert_eq!(individual.get_objectives(), &[0.4, 0.5]);
        
        // Test rank
        individual.set_rank(3);
        assert_eq!(individual.get_rank(), 3);
        
        // Test crowding distance
        individual.set_crowding_distance(1.5);
        assert_eq!(individual.get_crowding_distance(), 1.5);
    }

    #[test]
    fn test_create_random_individual() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.finalize();
        
        let _individual = create_random_individual(&graph, 3);
        
        // Can't make specific assertions due to randomness
        // Just ensure it doesn't panic
    }
}