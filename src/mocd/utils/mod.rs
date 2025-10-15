//! Utility functions and helper modules
//! 
//! This module provides various utilities for the community detection algorithms,
//! including convergence criteria, performance metrics, and threading utilities.

pub mod convergence;
pub mod metrics;
pub mod threading;

// Re-export commonly used items
pub use convergence::ConvergenceCriteria;
pub use metrics::{
    CommunityMetrics, 
    PerformanceMetrics, 
    MultiobjectiveMetrics, 
    ComprehensiveMetrics
};
pub use threading::{
    set_thread_count, 
    get_thread_count, 
    get_cpu_count, 
    get_thread_info,
    with_thread_count,
    optimal_thread_count,
    ThreadInfo
};

use crate::graph::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

/// Debug macro for conditional debug output
/// 
/// This macro provides flexible debug output based on debug levels and types.
/// It's used throughout the codebase for debugging information.
/// 
/// # Arguments
/// * `level` - Debug level (debug, warn, err)
/// * `fmt` - Format string
/// * `args` - Format arguments
#[macro_export]
macro_rules! debug {
    (debug, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[DEBUG] {}", format!($($arg)*));
    };
    (warn, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[WARN] {}", format!($($arg)*));
    };
    (err, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!("[ERROR] {}", format!($($arg)*));
    };
}

/// Normalizes community IDs to ensure they start from 0 and are consecutive
/// 
/// Isolated nodes (degree 0) are assigned community ID -1.
/// 
/// # Arguments
/// * `graph` - The graph being partitioned
/// * `partition` - The partition to normalize
/// 
/// # Returns
/// A new partition with normalized community IDs
pub fn normalize_community_ids(graph: &Graph, partition: Partition) -> Partition {
    let mut new_partition: FxHashMap<NodeId, CommunityId> = FxHashMap::default();
    let mut id_mapping: FxHashMap<CommunityId, CommunityId> = FxHashMap::default();
    let mut next_id: CommunityId = 0;

    for &node in graph.nodes.iter() {
        let is_isolated = match graph.adjacency_list.get(&node) {
            Some(neighbors) => neighbors.is_empty(),
            None => true, // if hasnt adjacency_list, it is isolated
        };

        if is_isolated {
            new_partition.insert(node, -1);
        } else {
            match partition.get(&node) {
                Some(&orig_comm) if orig_comm != -1 => {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        id_mapping.entry(orig_comm)
                    {
                        e.insert(next_id);
                        next_id += 1;
                    }
                    let mapped = *id_mapping.get(&orig_comm).unwrap();
                    new_partition.insert(node, mapped);
                }
                _ => {
                    new_partition.insert(node, -1);
                }
            }
        }
    }

    new_partition
}

/// Converts a Python dictionary to a Partition
/// 
/// # Arguments
/// * `py_dict` - Python dictionary with node->community mappings
/// 
/// # Returns
/// Partition representation as FxHashMap
pub fn to_partition(py_dict: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut part: FxHashMap<i32, i32> = FxHashMap::default();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}

/// Timer utility for measuring algorithm performance
pub struct Timer {
    start_time: std::time::Instant,
}

impl Timer {
    /// Creates and starts a new timer
    pub fn start() -> Self {
        Timer {
            start_time: std::time::Instant::now(),
        }
    }

    /// Gets elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    /// Gets elapsed time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Resets the timer
    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
    }
}

/// Memory usage estimation utilities
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimates memory usage for a graph
    pub fn graph_memory(graph: &crate::graph::Graph) -> usize {
        graph.memory_stats().total()
    }

    /// Estimates memory usage for a population of partitions
    pub fn population_memory(population_size: usize, num_nodes: usize) -> usize {
        // Rough estimate: each partition is a HashMap with node->community mappings
        let hashmap_overhead = 64; // Rough estimate for HashMap overhead
        let entry_size = std::mem::size_of::<(i32, i32)>();
        population_size * (hashmap_overhead + num_nodes * entry_size)
    }

    /// Gets current memory usage of the process (if available)
    pub fn current_memory_usage() -> Option<usize> {
        // This would require a system-specific implementation
        // For now, return None as a placeholder
        None
    }
}

/// Random number generation utilities
pub mod random {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    /// Creates a seeded random number generator for reproducible results
    pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// Creates a random number generator from entropy
    pub fn entropy_rng() -> ChaCha8Rng {
        use rand::SeedableRng;
        let mut rng = rand::rng();
        ChaCha8Rng::from_rng(&mut rng)
    }

    /// Generates a random partition for a graph
    pub fn random_partition(graph: &crate::graph::Graph, max_communities: usize) -> crate::graph::Partition {
        use rustc_hash::FxHashMap;
        
        let mut rng = entropy_rng();
        let mut partition = FxHashMap::default();
        
        for node in graph.nodes_iter() {
            let community = if graph.degree(node) == 0 {
                -1 // Isolated nodes get -1
            } else {
                rng.random_range(0..max_communities as i32)
            };
            partition.insert(*node, community);
        }
        
        partition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10);
        
        let elapsed_secs = timer.elapsed_secs();
        assert!(elapsed_secs >= 0.01);
    }

    #[test]
    fn test_memory_estimator() {
        // Test that functions don't panic and return reasonable values
        let memory = MemoryEstimator::population_memory(100, 1000);
        assert!(memory > 0);
        
        // Current memory usage might not be available on all systems
        let current = MemoryEstimator::current_memory_usage();
        // Just ensure it doesn't panic
        let _ = current;
    }

    #[test]
    fn test_random_utilities() {
        let _rng1 = random::seeded_rng(42);
        let _rng2 = random::seeded_rng(42);
        // Both should be seeded the same way (we can't easily test this without generating numbers)
        
        let _entropy_rng = random::entropy_rng();
        // Just ensure it doesn't panic
    }

    #[test]
    fn test_random_partition() {
        let mut graph = crate::graph::Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.finalize();
        
        let partition = random::random_partition(&graph, 3);
        
        // Should have all nodes
        assert_eq!(partition.len(), 3);
        assert!(partition.contains_key(&0));
        assert!(partition.contains_key(&1));
        assert!(partition.contains_key(&2));
        
        // All communities should be in valid range
        for &community in partition.values() {
            assert!(community >= 0 && community < 3);
        }
    }
}