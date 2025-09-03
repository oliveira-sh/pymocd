//! Convergence criteria for evolutionary algorithms
//!
//! This module provides tools for determining when an evolutionary algorithm
//! should terminate based on lack of improvement over multiple generations.

/// Represents the convergence criteria and state for genetic algorithms
///
/// Tracks the best fitness found so far and counts generations without improvement.
/// The algorithm is considered converged when no improvement is seen for a
/// specified number of generations.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    current_best_fitness: f64,       // Current best fitness value found
    generations_unchanged: usize,    // Number of generations without improvement
    max_stagnant_generations: usize, // Maximum allowed generations without improvement
    tolerance: f64,                  // Numerical tolerance for fitness comparison
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        ConvergenceCriteria {
            current_best_fitness: f64::MIN,
            generations_unchanged: 0,
            max_stagnant_generations: 100,
            tolerance: 1e-6,
        }
    }
}

impl ConvergenceCriteria {
    /// Creates new convergence criteria with custom parameters
    ///
    /// # Arguments
    /// * `max_stagnant_generations` - Maximum generations without improvement before convergence
    /// * `tolerance` - Minimum improvement required to reset stagnation counter
    ///
    /// # Returns
    /// New ConvergenceCriteria instance
    pub fn new(max_stagnant_generations: usize, tolerance: f64) -> Self {
        ConvergenceCriteria {
            current_best_fitness: f64::MIN,
            generations_unchanged: 0,
            max_stagnant_generations,
            tolerance,
        }
    }

    /// Checks if the algorithm has converged based on the latest fitness value
    ///
    /// # Arguments
    /// * `new_fitness` - Latest best fitness value from the current generation
    ///
    /// # Returns
    /// True if convergence criteria are met (algorithm should stop), false otherwise
    pub fn has_converged(&mut self, new_fitness: f64) -> bool {
        // Check if there's a significant improvement
        let improvement = new_fitness - self.current_best_fitness;
        let has_improved = improvement > self.tolerance;

        if has_improved {
            // Reset counter if we found a better solution
            self.current_best_fitness = new_fitness;
            self.generations_unchanged = 0;
            return false;
        }

        // No improvement, increment counter
        self.generations_unchanged += 1;

        // Check if we've exceeded the stagnation limit
        self.generations_unchanged >= self.max_stagnant_generations
    }

    /// Gets the current best fitness value
    pub fn get_best_fitness(&self) -> f64 {
        self.current_best_fitness
    }

    /// Gets the number of generations without improvement
    pub fn get_stagnant_generations(&self) -> usize {
        self.generations_unchanged
    }

    /// Gets the maximum allowed stagnant generations
    pub fn get_max_stagnant_generations(&self) -> usize {
        self.max_stagnant_generations
    }

    /// Gets the tolerance threshold for improvements
    pub fn get_tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Resets the convergence state (useful for algorithm restarts)
    pub fn reset(&mut self) {
        self.current_best_fitness = f64::MIN;
        self.generations_unchanged = 0;
    }

    /// Updates tolerance for dynamic convergence criteria
    ///
    /// # Arguments
    /// * `new_tolerance` - New tolerance value
    pub fn set_tolerance(&mut self, new_tolerance: f64) {
        self.tolerance = new_tolerance;
    }

    /// Updates maximum stagnant generations
    ///
    /// # Arguments
    /// * `max_gens` - New maximum stagnant generations
    pub fn set_max_stagnant_generations(&mut self, max_gens: usize) {
        self.max_stagnant_generations = max_gens;
    }

    /// Gets convergence progress as a percentage (0.0 to 1.0)
    ///
    /// # Returns
    /// Progress towards convergence, where 1.0 means converged
    pub fn convergence_progress(&self) -> f64 {
        if self.max_stagnant_generations == 0 {
            return 0.0;
        }

        (self.generations_unchanged as f64 / self.max_stagnant_generations as f64).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_criteria_default() {
        let criteria = ConvergenceCriteria::default();
        assert_eq!(criteria.get_max_stagnant_generations(), 100);
        assert_eq!(criteria.get_tolerance(), 1e-6);
        assert_eq!(criteria.get_stagnant_generations(), 0);
    }

    #[test]
    fn test_convergence_criteria_new() {
        let criteria = ConvergenceCriteria::new(50, 0.01);
        assert_eq!(criteria.get_max_stagnant_generations(), 50);
        assert_eq!(criteria.get_tolerance(), 0.01);
    }

    #[test]
    fn test_has_converged_with_improvement() {
        let mut criteria = ConvergenceCriteria::new(3, 0.1);

        // First fitness value
        assert!(!criteria.has_converged(1.0));
        assert_eq!(criteria.get_best_fitness(), 1.0);
        assert_eq!(criteria.get_stagnant_generations(), 0);

        // Improvement beyond tolerance
        assert!(!criteria.has_converged(1.2));
        assert_eq!(criteria.get_best_fitness(), 1.2);
        assert_eq!(criteria.get_stagnant_generations(), 0);
    }

    #[test]
    fn test_has_converged_without_improvement() {
        let mut criteria = ConvergenceCriteria::new(3, 0.1);

        // Set initial fitness
        criteria.has_converged(1.0);

        // No improvement (within tolerance)
        assert!(!criteria.has_converged(1.05)); // Gen 1
        assert_eq!(criteria.get_stagnant_generations(), 1);

        assert!(!criteria.has_converged(0.95)); // Gen 2
        assert_eq!(criteria.get_stagnant_generations(), 2);

        assert!(criteria.has_converged(1.02)); // Gen 3 - should converge
        assert_eq!(criteria.get_stagnant_generations(), 3);
    }

    #[test]
    fn test_reset() {
        let mut criteria = ConvergenceCriteria::new(3, 0.1);

        // Set some state
        criteria.has_converged(1.0);
        criteria.has_converged(0.9);

        assert_eq!(criteria.get_stagnant_generations(), 1);

        // Reset should clear state
        criteria.reset();
        assert_eq!(criteria.get_stagnant_generations(), 0);
        assert_eq!(criteria.get_best_fitness(), f64::MIN);
    }

    #[test]
    fn test_convergence_progress() {
        let mut criteria = ConvergenceCriteria::new(4, 0.1);

        criteria.has_converged(1.0);
        assert_eq!(criteria.convergence_progress(), 0.0);

        criteria.has_converged(0.9);
        assert_eq!(criteria.convergence_progress(), 0.25);

        criteria.has_converged(0.9);
        assert_eq!(criteria.convergence_progress(), 0.5);

        criteria.has_converged(0.9);
        assert_eq!(criteria.convergence_progress(), 0.75);

        criteria.has_converged(0.9);
        assert_eq!(criteria.convergence_progress(), 1.0);
    }

    #[test]
    fn test_setters() {
        let mut criteria = ConvergenceCriteria::default();

        criteria.set_tolerance(0.5);
        assert_eq!(criteria.get_tolerance(), 0.5);

        criteria.set_max_stagnant_generations(25);
        assert_eq!(criteria.get_max_stagnant_generations(), 25);
    }
}
