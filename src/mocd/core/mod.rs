//! Core algorithms for multiobjective community detection
//!
//! This module contains the main algorithm implementations, including the
//! high-performance MOCD (HP-MOCD) algorithm and deprecated legacy algorithms.

pub mod hpmocd;

// Re-export the main algorithm
pub use hpmocd::{HpMocd, Individual};

/// Algorithm configuration parameters
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Population size for the genetic algorithm
    pub population_size: usize,
    /// Number of generations to run
    pub num_generations: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Debug output level (0 = none, higher = more verbose)
    pub debug_level: i8,
    /// Tournament size for selection
    pub tournament_size: usize,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        AlgorithmConfig {
            population_size: 100,
            num_generations: 100,
            crossover_rate: 0.7,
            mutation_rate: 0.5,
            debug_level: 0,
            tournament_size: 2,
        }
    }
}

impl AlgorithmConfig {
    /// Creates a new configuration with custom parameters
    pub fn new(
        population_size: usize,
        num_generations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> Self {
        AlgorithmConfig {
            population_size,
            num_generations,
            crossover_rate,
            mutation_rate,
            debug_level: 0,
            tournament_size: 2,
        }
    }

    /// Sets debug level
    pub fn with_debug_level(mut self, debug_level: i8) -> Self {
        self.debug_level = debug_level;
        self
    }

    /// Sets tournament size
    pub fn with_tournament_size(mut self, tournament_size: usize) -> Self {
        self.tournament_size = tournament_size;
        self
    }

    /// Validates the configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size == 0 {
            return Err("Population size must be greater than 0".to_string());
        }

        if self.num_generations == 0 {
            return Err("Number of generations must be greater than 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.crossover_rate) {
            return Err("Crossover rate must be between 0.0 and 1.0".to_string());
        }

        if !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err("Mutation rate must be between 0.0 and 1.0".to_string());
        }

        if self.tournament_size == 0 {
            return Err("Tournament size must be greater than 0".to_string());
        }

        Ok(())
    }
}
