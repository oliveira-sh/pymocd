//! Graph data structures and operations for community detection
//! 
//! This module provides optimized graph representations and utilities specifically
//! designed for multiobjective community detection algorithms. It includes:
//! 
//! - High-performance graph data structure with adjacency lists
//! - Partition utilities for community assignments  
//! - Python interoperability for NetworkX and igraph
//! 
//! # Examples
//! 
//! ```rust
//! use pymocd::graph::{Graph, normalize_community_ids};
//! use rustc_hash::FxHashMap;
//! 
//! // Create a simple graph
//! let mut graph = Graph::new();
//! graph.add_edge(0, 1);
//! graph.add_edge(1, 2);
//! graph.finalize();
//! 
//! // Create a partition
//! let mut partition = FxHashMap::default();
//! partition.insert(0, 0);
//! partition.insert(1, 0);
//! partition.insert(2, 1);
//! 
//! // Normalize community IDs
//! let normalized = normalize_community_ids(&graph, partition);
//! ```

pub mod graph;
pub mod partition;
pub mod python_interop;

// Re-export commonly used types and functions
pub use graph::{Graph, NodeId, CommunityId, GraphMemoryStats};
pub use partition::{
    Partition, 
    normalize_community_ids, 
    calculate_modularity,
    validate_partition,
    get_node_community,
    get_community_nodes,
    count_communities
};
pub use python_interop::{
    get_nodes, 
    get_edges, 
    from_python_graph,
    is_supported_graph,
    get_graph_stats
};

impl Graph {
    /// Creates a Graph from a Python graph object (NetworkX or igraph)
    /// 
    /// This is a convenience method that wraps the python_interop functionality.
    /// 
    /// # Arguments
    /// * `pygraph` - Python graph object
    /// 
    /// # Returns
    /// A new Graph instance
    pub fn from_python(pygraph: &pyo3::Bound<'_, pyo3::PyAny>) -> Self {
        python_interop::from_python_graph(pygraph)
    }
}