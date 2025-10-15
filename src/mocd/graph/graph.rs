//! Core graph data structure for community detection algorithms
//! 
//! This module provides an optimized graph implementation specifically designed
//! for multiobjective community detection algorithms, with efficient adjacency
//! list representation and degree precomputation.
//!
//! # Examples
//! 
//! ```rust
//! use pymocd::graph::Graph;
//! 
//! let mut graph = Graph::new();
//! graph.add_edge(0, 1);
//! graph.add_edge(1, 2);
//! graph.finalize();
//! 
//! assert_eq!(graph.num_nodes(), 3);
//! assert_eq!(graph.num_edges(), 2);
//! ```

use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};


pub type NodeId = i32;
pub type CommunityId = i32;

/// Optimized graph structure for community detection algorithms
///
/// This graph implementation is specifically optimized for community detection
/// algorithms, providing fast neighbor lookups, degree queries, and edge existence checks.
#[derive(Debug, Clone)]
pub struct Graph {
    pub edges: Vec<(NodeId, NodeId)>,
    pub nodes: FxHashSet<NodeId>,
    pub adjacency_list: FxHashMap<NodeId, Vec<NodeId>>,
    pub degrees: FxHashMap<NodeId, usize>,
    pub node_vec: Vec<NodeId>,
    pub max_degree: usize,
    pub total_degree: usize,
    pub edge_lookup: FxHashSet<(NodeId, NodeId)>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates a new empty graph
    pub fn new() -> Self {
        Graph {
            edges: Vec::new(),
            nodes: FxHashSet::default(),
            adjacency_list: FxHashMap::default(),
            degrees: FxHashMap::default(),
            node_vec: Vec::new(),
            max_degree: 0,
            total_degree: 0,
            edge_lookup: FxHashSet::default(),
        }
    }

    /// Adds an undirected edge between two nodes
    /// 
    /// Self-loops are ignored. Duplicate edges are detected and ignored.
    /// 
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        if from == to {
            return; // Skip self-loops
        }

        let edge_key = if from < to { (from, to) } else { (to, from) };
        if self.edge_lookup.contains(&edge_key) {
            return; // Skip duplicate edges
        }

        self.edges.push((from, to));
        self.nodes.insert(from);
        self.nodes.insert(to);
        self.edge_lookup.insert(edge_key);

        self.adjacency_list.entry(from).or_default().push(to);
        self.adjacency_list.entry(to).or_default().push(from);

        let from_degree = self.adjacency_list[&from].len();
        let to_degree = self.adjacency_list[&to].len();

        self.degrees.insert(from, from_degree);
        self.degrees.insert(to, to_degree);

        self.max_degree = self.max_degree.max(from_degree).max(to_degree);
        self.total_degree += 2;
    }

    /// Finalizes the graph structure for optimal performance
    /// 
    /// This method should be called after all edges have been added.
    /// It optimizes the internal data structures for efficient access.
    pub fn finalize(&mut self) {
        self.node_vec = self.nodes.iter().copied().collect();
        self.node_vec.sort_unstable();

        // Shrink collections to save memory
        self.degrees.shrink_to_fit();
        self.edge_lookup.shrink_to_fit();

        // Sort adjacency lists for consistent iteration and shrink to fit
        for neighbors in self.adjacency_list.values_mut() {
            neighbors.sort_unstable();
            neighbors.shrink_to_fit();
        }
    }

    /// Creates a graph from an adjacency list file
    /// 
    /// # File Format
    /// Each line should contain a node ID followed by its neighbors, separated by whitespace.
    /// Lines starting with '#' are treated as comments and ignored.
    /// 
    /// # Arguments
    /// * `file_path` - Path to the adjacency list file
    pub fn from_adj_list(file_path: &str) -> Self {
        let mut graph = Graph::new();
        let file = File::open(file_path).expect("Unable to open file");
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.expect("Could not read line");
            let parts: Vec<&str> = line.split_whitespace().collect();

            if line.trim().starts_with('#') || parts.is_empty() {
                continue;
            }

            let node: NodeId = parts[0].parse().expect("First item should be node ID");
            for neighbor_str in &parts[1..] {
                let neighbor: NodeId = neighbor_str
                    .parse()
                    .expect("Neighbor should be a valid node ID");
                graph.add_edge(node, neighbor);
            }
        }

        graph.finalize();
        graph
    }

    /// Prints graph statistics for debugging
    pub fn print(&self) {
        crate::debug!(
            debug,
            "G = ({},{}) | Max Degree: {} | Avg Degree: {:.2}",
            self.num_nodes(),
            self.num_edges(),
            self.max_degree,
            self.avg_degree()
        );
    }

    // Query methods

    /// Returns the neighbors of a given node
    /// 
    /// # Arguments
    /// * `node` - Node ID to get neighbors for
    /// 
    /// # Returns
    /// Slice of neighbor node IDs, or empty slice if node doesn't exist
    #[inline(always)]
    pub fn neighbors(&self, node: &NodeId) -> &[NodeId] {
        self.adjacency_list.get(node).map_or(&[], |x| x)
    }

    /// Returns the degree of a given node
    /// 
    /// # Arguments
    /// * `node` - Node ID to get degree for
    /// 
    /// # Returns
    /// Degree of the node, or 0 if node doesn't exist
    #[inline(always)]
    pub fn degree(&self, node: &NodeId) -> usize {
        *self.degrees.get(node).unwrap_or(&0)
    }

    /// Checks if an edge exists between two nodes
    /// 
    /// # Arguments
    /// * `from` - First node ID
    /// * `to` - Second node ID
    /// 
    /// # Returns
    /// True if edge exists, false otherwise
    #[inline(always)]
    pub fn has_edge(&self, from: NodeId, to: NodeId) -> bool {
        let edge_key = if from < to { (from, to) } else { (to, from) };
        self.edge_lookup.contains(&edge_key)
    }

    /// Returns an iterator over all nodes
    #[inline(always)]
    pub fn nodes_iter(&self) -> impl Iterator<Item = &NodeId> {
        self.node_vec.iter()
    }

    /// Returns a reference to the sorted vector of nodes
    #[inline(always)]
    pub fn nodes_vec(&self) -> &Vec<NodeId> {
        &self.node_vec
    }

    /// Returns the number of nodes in the graph
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph
    #[inline(always)]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns precomputed degrees for all nodes
    #[inline(always)]
    pub fn precompute_degrees(&self) -> &FxHashMap<NodeId, usize> {
        &self.degrees
    }

    /// Returns the maximum degree in the graph
    #[inline(always)]
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the total degree (sum of all node degrees)
    #[inline(always)]
    pub fn total_degree(&self) -> usize {
        self.total_degree
    }

    /// Returns the average degree of the graph
    #[inline(always)]
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.total_degree as f64 / self.num_nodes() as f64
        }
    }

    /// Returns memory usage statistics for the graph
    pub fn memory_stats(&self) -> GraphMemoryStats {
        GraphMemoryStats {
            nodes_memory: self.nodes.len() * std::mem::size_of::<NodeId>(),
            edges_memory: self.edges.len() * std::mem::size_of::<(NodeId, NodeId)>(),
            adjacency_memory: self
                .adjacency_list
                .values()
                .map(|v| v.capacity() * std::mem::size_of::<NodeId>())
                .sum(),
            degrees_memory: self.degrees.len()
                * (std::mem::size_of::<NodeId>() + std::mem::size_of::<usize>()),
            edge_lookup_memory: self.edge_lookup.len() * std::mem::size_of::<(NodeId, NodeId)>(),
        }
    }
}

/// Memory usage statistics for a graph
pub struct GraphMemoryStats {
    pub nodes_memory: usize,
    pub edges_memory: usize,
    pub adjacency_memory: usize,
    pub degrees_memory: usize,
    pub edge_lookup_memory: usize,
}

impl GraphMemoryStats {
    /// Returns total memory usage in bytes
    pub fn total(&self) -> usize {
        self.nodes_memory
            + self.edges_memory
            + self.adjacency_memory
            + self.degrees_memory
            + self.edge_lookup_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.finalize();

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0));
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.max_degree(), 3);
        assert_eq!(graph.total_degree(), 6);
        assert_eq!(graph.avg_degree(), 1.5);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        let mut neighbors: Vec<NodeId> = graph.neighbors(&0).to_vec();
        neighbors.sort();
        assert_eq!(neighbors, [1, 2, 4]);
    }

    #[test]
    fn test_degree_queries() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        assert_eq!(graph.degree(&0), 3);
        assert_eq!(graph.degree(&1), 1);
        assert_eq!(graph.degree(&2), 1);
        assert_eq!(graph.degree(&4), 1);
        assert_eq!(graph.degree(&999), 0); // Non-existent node
    }

    #[test]
    fn test_duplicate_edge_prevention() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 0); // Should be ignored (same edge)
        graph.add_edge(0, 1); // Should be ignored (duplicate)
        graph.finalize();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.degree(&0), 1);
        assert_eq!(graph.degree(&1), 1);
    }

    #[test]
    fn test_self_loop_prevention() {
        let mut graph = Graph::new();
        graph.add_edge(0, 0); // Should be ignored
        graph.add_edge(0, 1);
        graph.finalize();

        assert_eq!(graph.num_edges(), 1);
        assert!(!graph.has_edge(0, 0));
        assert!(graph.has_edge(0, 1));
    }

    #[test]
    fn test_nodes_iteration() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);
        graph.finalize();

        let mut nodes: Vec<NodeId> = graph.nodes_iter().copied().collect();
        nodes.sort();
        assert_eq!(nodes, [0, 1, 2, 3]);
    }
}