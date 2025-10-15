//! Performance metrics and evaluation utilities
//! 
//! This module provides various metrics for evaluating community detection quality
//! and algorithm performance.

use crate::graph::{Graph, Partition};
use std::collections::HashMap;

/// Community detection quality metrics
#[derive(Debug, Clone)]
pub struct CommunityMetrics {
    /// Modularity score (higher is better for community detection)
    pub modularity: f64,
    /// Number of communities detected
    pub num_communities: usize,
    /// Number of isolated nodes (degree 0)
    pub isolated_nodes: usize,
    /// Average community size
    pub avg_community_size: f64,
    /// Size of the largest community
    pub max_community_size: usize,
    /// Size of the smallest community (excluding isolated nodes)
    pub min_community_size: usize,
}

/// Algorithm performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total runtime in milliseconds
    pub runtime_ms: u128,
    /// Number of generations/iterations executed
    pub generations: usize,
    /// Final population size
    pub population_size: usize,
    /// Memory usage in bytes (approximate)
    pub memory_bytes: usize,
    /// Convergence generation (when algorithm converged, if applicable)
    pub convergence_generation: Option<usize>,
}

/// Multiobjective optimization metrics for MOCD algorithms
#[derive(Debug, Clone)]
pub struct MultiobjectiveMetrics {
    /// Number of solutions in the Pareto front
    pub pareto_front_size: usize,
    /// Hypervolume indicator (if reference point provided)
    pub hypervolume: Option<f64>,
    /// Spread/diversity of the Pareto front
    pub diversity: f64,
    /// Average distance between adjacent solutions
    pub spacing: f64,
}

impl CommunityMetrics {
    /// Calculates comprehensive community detection metrics
    /// 
    /// # Arguments
    /// * `graph` - The graph being analyzed
    /// * `partition` - The community partition to evaluate
    /// 
    /// # Returns
    /// CommunityMetrics containing various quality measures
    pub fn calculate(graph: &Graph, partition: &Partition) -> Self {
        let modularity = crate::graph::calculate_modularity(graph, partition);
        let community_sizes = Self::calculate_community_sizes(partition);
        
        let num_communities = community_sizes.len();
        let isolated_nodes = partition.values().filter(|&&comm| comm == -1).count();
        
        let total_non_isolated: usize = community_sizes.values().sum();
        let avg_community_size = if num_communities > 0 {
            total_non_isolated as f64 / num_communities as f64
        } else {
            0.0
        };
        
        let max_community_size = community_sizes.values().max().copied().unwrap_or(0);
        let min_community_size = community_sizes.values().min().copied().unwrap_or(0);

        CommunityMetrics {
            modularity,
            num_communities,
            isolated_nodes,
            avg_community_size,
            max_community_size,
            min_community_size,
        }
    }

    /// Calculates the size of each community
    fn calculate_community_sizes(partition: &Partition) -> HashMap<i32, usize> {
        let mut sizes = HashMap::new();
        
        for &community in partition.values() {
            if community != -1 { // Exclude isolated nodes
                *sizes.entry(community).or_insert(0) += 1;
            }
        }
        
        sizes
    }

    /// Returns a quality score combining multiple metrics
    /// 
    /// This is a simple composite score that can be used for comparison.
    /// Higher values indicate better community structure.
    pub fn quality_score(&self) -> f64 {
        // Simple weighted combination - can be customized based on requirements
        let modularity_weight = 0.7;
        let size_balance_weight = 0.3;
        
        let modularity_component = self.modularity.max(0.0);
        
        // Penalize extreme community size imbalances
        let size_balance = if self.num_communities > 1 {
            let size_ratio = self.min_community_size as f64 / self.max_community_size as f64;
            size_ratio.min(1.0).max(0.0)
        } else {
            1.0
        };
        
        modularity_weight * modularity_component + size_balance_weight * size_balance
    }
}

impl PerformanceMetrics {
    /// Creates new performance metrics
    pub fn new() -> Self {
        PerformanceMetrics {
            runtime_ms: 0,
            generations: 0,
            population_size: 0,
            memory_bytes: 0,
            convergence_generation: None,
        }
    }

    /// Updates runtime measurement
    pub fn set_runtime(&mut self, runtime_ms: u128) {
        self.runtime_ms = runtime_ms;
    }

    /// Updates generation count
    pub fn set_generations(&mut self, generations: usize) {
        self.generations = generations;
    }

    /// Sets convergence information
    pub fn set_convergence(&mut self, generation: usize) {
        self.convergence_generation = Some(generation);
    }

    /// Calculates generations per second
    pub fn generations_per_second(&self) -> f64 {
        if self.runtime_ms == 0 {
            0.0
        } else {
            self.generations as f64 / (self.runtime_ms as f64 / 1000.0)
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiobjectiveMetrics {
    /// Creates new multiobjective metrics
    pub fn new() -> Self {
        MultiobjectiveMetrics {
            pareto_front_size: 0,
            hypervolume: None,
            diversity: 0.0,
            spacing: 0.0,
        }
    }

    /// Calculates metrics for a Pareto front
    /// 
    /// # Arguments
    /// * `objectives` - Vector of objective vectors for each solution
    /// 
    /// # Returns
    /// MultiobjectiveMetrics for the given Pareto front
    pub fn calculate_for_front(objectives: &[[f64; 2]]) -> Self {
        let pareto_front_size = objectives.len();
        
        if pareto_front_size == 0 {
            return Self::new();
        }

        let diversity = Self::calculate_diversity(objectives);
        let spacing = Self::calculate_spacing(objectives);

        MultiobjectiveMetrics {
            pareto_front_size,
            hypervolume: None, // Can be implemented if reference point is known
            diversity,
            spacing,
        }
    }

    /// Calculates the diversity (spread) of solutions in objective space
    fn calculate_diversity(objectives: &[[f64; 2]]) -> f64 {
        if objectives.len() < 2 {
            return 0.0;
        }

        // Calculate ranges for each objective
        let mut min_obj1 = f64::INFINITY;
        let mut max_obj1 = f64::NEG_INFINITY;
        let mut min_obj2 = f64::INFINITY;
        let mut max_obj2 = f64::NEG_INFINITY;

        for obj in objectives {
            min_obj1 = min_obj1.min(obj[0]);
            max_obj1 = max_obj1.max(obj[0]);
            min_obj2 = min_obj2.min(obj[1]);
            max_obj2 = max_obj2.max(obj[1]);
        }

        let range1 = max_obj1 - min_obj1;
        let range2 = max_obj2 - min_obj2;

        // Diversity is the diagonal of the bounding box
        (range1.powi(2) + range2.powi(2)).sqrt()
    }

    /// Calculates the spacing (uniformity) of solutions in objective space
    fn calculate_spacing(objectives: &[[f64; 2]]) -> f64 {
        if objectives.len() < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();
        
        for i in 0..objectives.len() {
            let mut min_distance = f64::INFINITY;
            
            for j in 0..objectives.len() {
                if i != j {
                    let dist = Self::euclidean_distance(&objectives[i], &objectives[j]);
                    min_distance = min_distance.min(dist);
                }
            }
            
            distances.push(min_distance);
        }

        // Calculate standard deviation of nearest neighbor distances
        let mean_distance: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance: f64 = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;

        variance.sqrt()
    }

    /// Calculates Euclidean distance between two objective vectors
    fn euclidean_distance(obj1: &[f64; 2], obj2: &[f64; 2]) -> f64 {
        ((obj1[0] - obj2[0]).powi(2) + (obj1[1] - obj2[1]).powi(2)).sqrt()
    }
}

impl Default for MultiobjectiveMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined metrics for comprehensive algorithm evaluation
#[derive(Debug, Clone)]
pub struct ComprehensiveMetrics {
    pub community: CommunityMetrics,
    pub performance: PerformanceMetrics,
    pub multiobjective: MultiobjectiveMetrics,
}

impl ComprehensiveMetrics {
    /// Creates comprehensive metrics for a community detection result
    pub fn calculate(
        graph: &Graph,
        partition: &Partition,
        objectives: &[[f64; 2]],
        performance: PerformanceMetrics,
    ) -> Self {
        let community = CommunityMetrics::calculate(graph, partition);
        let multiobjective = MultiobjectiveMetrics::calculate_for_front(objectives);

        ComprehensiveMetrics {
            community,
            performance,
            multiobjective,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, Partition};
    use rustc_hash::FxHashMap;

    fn create_test_graph_and_partition() -> (Graph, Partition) {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0); // Triangle
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 3); // Another triangle
        graph.finalize();

        let mut partition = FxHashMap::default();
        partition.insert(0, 0);
        partition.insert(1, 0);
        partition.insert(2, 0);
        partition.insert(3, 1);
        partition.insert(4, 1);
        partition.insert(5, 1);

        (graph, partition)
    }

    #[test]
    fn test_community_metrics() {
        let (graph, partition) = create_test_graph_and_partition();
        let metrics = CommunityMetrics::calculate(&graph, &partition);

        assert_eq!(metrics.num_communities, 2);
        assert_eq!(metrics.isolated_nodes, 0);
        assert_eq!(metrics.avg_community_size, 3.0);
        assert_eq!(metrics.max_community_size, 3);
        assert_eq!(metrics.min_community_size, 3);
        assert!(metrics.modularity > 0.0); // Should have positive modularity
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.set_runtime(2000); // 2 seconds
        metrics.set_generations(100);
        metrics.set_convergence(80);

        assert_eq!(metrics.runtime_ms, 2000);
        assert_eq!(metrics.generations, 100);
        assert_eq!(metrics.convergence_generation, Some(80));
        assert_eq!(metrics.generations_per_second(), 50.0);
    }

    #[test]
    fn test_multiobjective_metrics() {
        let objectives = vec![
            [0.1, 0.9],
            [0.3, 0.7],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.9, 0.1],
        ];

        let metrics = MultiobjectiveMetrics::calculate_for_front(&objectives);
        
        assert_eq!(metrics.pareto_front_size, 5);
        assert!(metrics.diversity > 0.0);
        assert!(metrics.spacing >= 0.0);
    }

    #[test]
    fn test_empty_objectives() {
        let objectives: Vec<[f64; 2]> = vec![];
        let metrics = MultiobjectiveMetrics::calculate_for_front(&objectives);
        
        assert_eq!(metrics.pareto_front_size, 0);
        assert_eq!(metrics.diversity, 0.0);
        assert_eq!(metrics.spacing, 0.0);
    }

    #[test]
    fn test_quality_score() {
        let (graph, partition) = create_test_graph_and_partition();
        let metrics = CommunityMetrics::calculate(&graph, &partition);
        
        let score = metrics.quality_score();
        assert!(score >= 0.0 && score <= 1.0);
    }
}