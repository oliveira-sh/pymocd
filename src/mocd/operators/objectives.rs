//! Objective functions for multiobjective community detection
//! 
//! This module implements the core objective functions used in the MOCD algorithms.
//! The objectives typically include measures of intra-community density and 
//! inter-community sparsity.

use crate::graph::{Graph, Partition, NodeId};
use rustc_hash::FxBuildHasher;
use rayon::prelude::*;
use std::collections::HashMap;

/// Metrics computed for a partition
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Intra-community edge density (lower is better for community structure)
    pub intra: f64,
    /// Inter-community edge density (lower is better for community structure)  
    pub inter: f64,
}

impl Metrics {
    /// Creates new metrics with specified values
    pub fn new(intra: f64, inter: f64) -> Self {
        Metrics { intra, inter }
    }

    /// Creates metrics with zero values
    pub fn zero() -> Self {
        Metrics::new(0.0, 0.0)
    }

    /// Calculates modularity from intra and inter metrics
    /// 
    /// Modularity Q = 1 - inter - intra for normalized metrics
    pub fn modularity(&self) -> f64 {
        1.0 - self.inter - self.intra
    }

    /// Returns true if this solution dominates the other in Pareto sense
    /// 
    /// For community detection, we want to minimize both intra and inter values.
    pub fn dominates(&self, other: &Metrics) -> bool {
        (self.intra <= other.intra && self.inter <= other.inter) &&
        (self.intra < other.intra || self.inter < other.inter)
    }

    /// Returns true if this solution is dominated by the other
    pub fn is_dominated_by(&self, other: &Metrics) -> bool {
        other.dominates(self)
    }
}

/// Calculates objective functions for a given partition
/// 
/// This is the core function that evaluates the quality of a community partition
/// using two objectives: intra-community and inter-community edge densities.
/// 
/// # Arguments
/// * `graph` - The graph being partitioned
/// * `partition` - The community assignment for each node
/// * `degrees` - Precomputed node degrees for efficiency
/// * `parallel` - Whether to use parallel computation
/// 
/// # Returns
/// Metrics containing the intra and inter objective values
pub fn calculate_objectives(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    parallel: bool,
) -> Metrics {
    if graph.num_edges() == 0 {
        return Metrics::zero();
    }

    let total_edges = graph.num_edges() as f64;
    
    if parallel {
        calculate_objectives_parallel(graph, partition, degrees, total_edges)
    } else {
        calculate_objectives_sequential(graph, partition, degrees, total_edges)
    }
}

/// Sequential implementation of objective calculation
fn calculate_objectives_sequential(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    total_edges: f64,
) -> Metrics {
    let mut intra_edges = 0;
    let mut inter_edges = 0;
    let mut intra_expected = 0.0;
    let mut inter_expected = 0.0;

    // Count actual edges and calculate expected values
    for &(from, to) in &graph.edges {
        let comm_from = partition.get(&from).copied().unwrap_or(-1);
        let comm_to = partition.get(&to).copied().unwrap_or(-1);

        // Skip isolated nodes
        if comm_from == -1 || comm_to == -1 {
            continue;
        }

        let deg_from = degrees.get(&from).copied().unwrap_or(0) as f64;
        let deg_to = degrees.get(&to).copied().unwrap_or(0) as f64;
        let expected = (deg_from * deg_to) / (2.0 * total_edges);

        if comm_from == comm_to {
            intra_edges += 1;
            intra_expected += expected;
        } else {
            inter_edges += 1;
            inter_expected += expected;
        }
    }

    // Normalize by total edges to get density-like measures
    let intra_actual = intra_edges as f64 / total_edges;
    let inter_actual = inter_edges as f64 / total_edges;
    let intra_exp_norm = intra_expected / total_edges;
    let inter_exp_norm = inter_expected / total_edges;

    // Objectives: deviation from null model (configuration model)
    let intra_obj = (intra_actual - intra_exp_norm).abs();
    let inter_obj = (inter_actual - inter_exp_norm).abs();

    Metrics::new(intra_obj, inter_obj)
}

/// Parallel implementation of objective calculation
fn calculate_objectives_parallel(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    total_edges: f64,
) -> Metrics {
    let (intra_data, inter_data): (Vec<_>, Vec<_>) = graph.edges
        .par_iter()
        .filter_map(|&(from, to)| {
            let comm_from = partition.get(&from).copied().unwrap_or(-1);
            let comm_to = partition.get(&to).copied().unwrap_or(-1);

            // Skip isolated nodes
            if comm_from == -1 || comm_to == -1 {
                return None;
            }

            let deg_from = degrees.get(&from).copied().unwrap_or(0) as f64;
            let deg_to = degrees.get(&to).copied().unwrap_or(0) as f64;
            let expected = (deg_from * deg_to) / (2.0 * total_edges);

            if comm_from == comm_to {
                Some((1, expected, 0, 0.0)) // intra edge
            } else {
                Some((0, 0.0, 1, expected)) // inter edge
            }
        })
        .partition(|&(intra_count, _, _inter_count, _)| intra_count > 0);

    let (intra_edges, intra_expected): (i32, f64) = intra_data
        .into_iter()
        .fold((0, 0.0), |acc, (count, exp, _, _)| (acc.0 + count, acc.1 + exp));

    let (inter_edges, inter_expected): (i32, f64) = inter_data
        .into_iter()
        .fold((0, 0.0), |acc, (_, _, count, exp)| (acc.0 + count, acc.1 + exp));

    // Normalize and calculate objectives
    let intra_actual = intra_edges as f64 / total_edges;
    let inter_actual = inter_edges as f64 / total_edges;
    let intra_exp_norm = intra_expected / total_edges;
    let inter_exp_norm = inter_expected / total_edges;

    let intra_obj = (intra_actual - intra_exp_norm).abs();
    let inter_obj = (inter_actual - inter_exp_norm).abs();

    Metrics::new(intra_obj, inter_obj)
}

/// Calculates modularity directly from a partition
/// 
/// This is a convenience function that wraps the objective calculation
/// and computes modularity.
/// 
/// # Arguments
/// * `graph` - The graph being analyzed
/// * `partition` - The community partition
/// * `degrees` - Precomputed node degrees
/// 
/// # Returns
/// The modularity value of the partition
pub fn calculate_modularity(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> f64 {
    let metrics = calculate_objectives(graph, partition, degrees, false);
    metrics.modularity()
}

/// Validates that objectives are properly bounded
/// 
/// Objectives should typically be in the range [0, 1] for normalized metrics.
/// 
/// # Arguments
/// * `metrics` - The metrics to validate
/// 
/// # Returns
/// True if metrics are within expected bounds
pub fn validate_metrics(metrics: &Metrics) -> bool {
    metrics.intra >= 0.0 && metrics.intra <= 1.0 &&
    metrics.inter >= 0.0 && metrics.inter <= 1.0
}

/// Calculates the weighted sum of objectives
/// 
/// This can be used for single-objective optimization or scalarization.
/// 
/// # Arguments
/// * `metrics` - The metrics to combine
/// * `intra_weight` - Weight for the intra-community objective
/// * `inter_weight` - Weight for the inter-community objective
/// 
/// # Returns
/// Weighted sum of the objectives
pub fn weighted_sum(metrics: &Metrics, intra_weight: f64, inter_weight: f64) -> f64 {
    intra_weight * metrics.intra + inter_weight * metrics.inter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use rustc_hash::FxHashMap;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();
        // Create two triangles connected by one edge
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0); // First triangle
        graph.add_edge(2, 3); // Bridge
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 3); // Second triangle
        graph.finalize();
        graph
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new(0.3, 0.4);
        assert_eq!(metrics.intra, 0.3);
        assert_eq!(metrics.inter, 0.4);

        let zero = Metrics::zero();
        assert_eq!(zero.intra, 0.0);
        assert_eq!(zero.inter, 0.0);
    }

    #[test]
    fn test_modularity_calculation() {
        let metrics = Metrics::new(0.2, 0.3);
        let modularity = metrics.modularity();
        assert!((modularity - 0.5).abs() < 1e-10); // 1.0 - 0.2 - 0.3
    }

    #[test]
    fn test_dominance() {
        let metrics1 = Metrics::new(0.2, 0.3);
        let metrics2 = Metrics::new(0.3, 0.4);
        let metrics3 = Metrics::new(0.2, 0.4);

        assert!(metrics1.dominates(&metrics2)); // Better in both objectives
        assert!(metrics1.dominates(&metrics3)); // Better in one, equal in other
        assert!(!metrics2.dominates(&metrics1));
        assert!(!metrics3.dominates(&metrics1));
    }

    #[test]
    fn test_objective_calculation() {
        let graph = create_test_graph();
        let mut partition = FxHashMap::default();
        
        // Perfect community structure: two separate communities
        partition.insert(0, 0);
        partition.insert(1, 0);
        partition.insert(2, 0);
        partition.insert(3, 1);
        partition.insert(4, 1);
        partition.insert(5, 1);

        let degrees = graph.precompute_degrees().clone();
        let metrics = calculate_objectives(&graph, &partition, &degrees, false);
        
        // Should have reasonable objective values
        assert!(validate_metrics(&metrics));
        assert!(metrics.intra >= 0.0);
        assert!(metrics.inter >= 0.0);
    }

    #[test]
    fn test_parallel_vs_sequential() {
        let graph = create_test_graph();
        let mut partition = FxHashMap::default();
        
        for node in graph.nodes_iter() {
            partition.insert(*node, *node % 2); // Alternate communities
        }

        let degrees = graph.precompute_degrees().clone();
        let sequential = calculate_objectives(&graph, &partition, &degrees, false);
        let parallel = calculate_objectives(&graph, &partition, &degrees, true);
        
        // Results should be the same (within floating point precision)
        let epsilon = 1e-10;
        assert!((sequential.intra - parallel.intra).abs() < epsilon);
        assert!((sequential.inter - parallel.inter).abs() < epsilon);
    }

    #[test]
    fn test_weighted_sum() {
        let metrics = Metrics::new(0.3, 0.4);
        let sum = weighted_sum(&metrics, 0.6, 0.4);
        assert_eq!(sum, 0.6 * 0.3 + 0.4 * 0.4);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::new();
        let partition = FxHashMap::default();
        let degrees = graph.precompute_degrees().clone();
        
        let metrics = calculate_objectives(&graph, &partition, &degrees, false);
        assert_eq!(metrics.intra, 0.0);
        assert_eq!(metrics.inter, 0.0);
    }
}