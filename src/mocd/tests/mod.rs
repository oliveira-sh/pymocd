//! Integration and unit tests for the PyMOCD library
//! 
//! This module contains comprehensive tests for all components of the
//! multiobjective community detection library.

pub mod graph_tests;

/// Common test utilities and helper functions
pub mod common {
    use crate::graph::Graph;
    use rustc_hash::FxHashMap;

    /// Creates a simple test graph with known structure
    /// 
    /// Returns a graph with two triangular communities connected by one edge:
    /// Triangle 1: nodes 0-1-2
    /// Triangle 2: nodes 3-4-5  
    /// Bridge: edge 2-3
    pub fn create_two_triangle_graph() -> Graph {
        let mut graph = Graph::new();
        
        // First triangle
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        
        // Bridge
        graph.add_edge(2, 3);
        
        // Second triangle
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 3);
        
        graph.finalize();
        graph
    }

    /// Creates a perfect partition for the two triangle graph
    pub fn create_perfect_partition() -> crate::graph::Partition {
        let mut partition = FxHashMap::default();
        partition.insert(0, 0);
        partition.insert(1, 0);
        partition.insert(2, 0);
        partition.insert(3, 1);
        partition.insert(4, 1);
        partition.insert(5, 1);
        partition
    }

    /// Creates a karate club-like graph for testing
    pub fn create_karate_club_graph() -> Graph {
        let mut graph = Graph::new();
        
        // Add edges representing a simplified version of Zachary's karate club
        let edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
            (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7),
            (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9),
            (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33),
            (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32),
            (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31),
            (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
        ];
        
        for &(from, to) in &edges {
            graph.add_edge(from, to);
        }
        
        graph.finalize();
        graph
    }

    /// Creates a random graph with specified parameters
    pub fn create_random_graph(num_nodes: usize, edge_probability: f64, seed: u64) -> Graph {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut graph = Graph::new();
        
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                if rng.random::<f64>() < edge_probability {
                    graph.add_edge(i as i32, j as i32);
                }
            }
        }
        
        graph.finalize();
        graph
    }

    /// Validates that a partition is properly formed
    pub fn validate_partition_structure(
        graph: &Graph, 
        partition: &crate::graph::Partition
    ) -> bool {
        // Check that all graph nodes are in the partition
        for node in graph.nodes_iter() {
            if !partition.contains_key(node) {
                return false;
            }
        }
        
        // Check that all partition nodes exist in the graph
        for &node in partition.keys() {
            if !graph.nodes.contains(&node) {
                return false;
            }
        }
        
        true
    }

    /// Calculates the number of intra-community edges
    pub fn count_intra_community_edges(
        graph: &Graph,
        partition: &crate::graph::Partition,
    ) -> usize {
        let mut count = 0;
        
        for &(from, to) in &graph.edges {
            if let (Some(&comm_from), Some(&comm_to)) = (partition.get(&from), partition.get(&to)) {
                if comm_from == comm_to && comm_from != -1 {
                    count += 1;
                }
            }
        }
        
        count
    }

    /// Calculates the number of inter-community edges
    pub fn count_inter_community_edges(
        graph: &Graph,
        partition: &crate::graph::Partition,
    ) -> usize {
        let mut count = 0;
        
        for &(from, to) in &graph.edges {
            if let (Some(&comm_from), Some(&comm_to)) = (partition.get(&from), partition.get(&to)) {
                if comm_from != comm_to && comm_from != -1 && comm_to != -1 {
                    count += 1;
                }
            }
        }
        
        count
    }
}