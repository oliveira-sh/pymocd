//! Python interoperability for NetworkX and igraph compatibility
//!
//! This module provides functionality to convert Python graph objects (NetworkX and igraph)
//! into the internal Graph representation used by the community detection algorithms.

use super::graph::{Graph, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Extracts nodes from a Python graph object
///
/// Supports both NetworkX and igraph Graph objects.
///
/// # Arguments
/// * `graph` - Python graph object (NetworkX or igraph)
///
/// # Returns
/// Vector of node IDs as integers
///
/// # Errors
/// Returns PyErr if the graph format is unsupported or node IDs are not integers
pub fn get_nodes(graph: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
    // Try NetworkX format first
    if let Ok(nx_nodes) = graph.call_method0("nodes") {
        let mut nodes: Vec<NodeId> = Vec::new();
        for node_obj_result in nx_nodes.try_iter()? {
            let node_obj = node_obj_result?;
            let node_id = match node_obj.extract::<i64>() {
                Ok(int_val) => int_val as NodeId,
                Err(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Failed getting node id's. Verify if all Graph.nodes are positive integers; <str> as node_id isn't supported",
                    ));
                }
            };
            nodes.push(node_id);
        }
        return Ok(nodes);
    }

    // Try igraph format
    if let Ok(vs) = graph.getattr("vs") {
        let iter_vs = vs.call_method0("__iter__")?;
        let mut nodes: Vec<NodeId> = Vec::new();

        for vertex_obj in iter_vs.try_iter()? {
            let vertex: Bound<'_, PyAny> = vertex_obj?;
            let index: NodeId = vertex.getattr("index")?.extract()?;
            nodes.push(index);
        }
        return Ok(nodes);
    }

    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Unable to get node list from NetworkX or igraph",
    ))
}

/// Extracts edges from a Python graph object
///
/// Supports both NetworkX and igraph Graph objects.
///
/// # Arguments
/// * `graph` - Python graph object (NetworkX or igraph)
///
/// # Returns
/// Vector of edge tuples (from_node, to_node)
///
/// # Errors
/// Returns PyErr if the graph format is unsupported
pub fn get_edges(graph: &Bound<'_, PyAny>) -> PyResult<Vec<(NodeId, NodeId)>> {
    let edges_iter = match graph.call_method0("edges") {
        Ok(nx_edges) => {
            // NetworkX format
            nx_edges.call_method0("__iter__")?
        }
        Err(_) => {
            crate::debug!(warn, "networkx.Graph() not found, trying igraph.Graph()");
            match graph.call_method0("get_edgelist") {
                Ok(ig_edges) => {
                    // igraph format
                    ig_edges.call_method0("__iter__")?
                }
                Err(_) => {
                    crate::debug!(err, "supported graph libraries not found");
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "neither NetworkX nor igraph graph methods are available",
                    ));
                }
            }
        }
    };

    let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
    for edge_obj in edges_iter.try_iter()? {
        let edge: Bound<'_, PyAny> = edge_obj?;
        let from: NodeId = edge.get_item(0)?.extract()?;
        let to: NodeId = edge.get_item(1)?.extract()?;
        edges.push((from, to));
    }

    Ok(edges)
}

/// Creates a Graph from a Python graph object
///
/// This is the main entry point for converting Python graph objects
/// (NetworkX or igraph) into the internal Graph representation.
///
/// # Arguments
/// * `pygraph` - Python graph object
///
/// # Returns
/// A new Graph instance
///
/// # Panics
/// Panics if the graph conversion fails (should be rare with valid input)
///
/// # Examples
///
/// ```python
/// import networkx as nx
/// G = nx.karate_club_graph()
/// # Pass G to Rust function that calls from_python_graph
/// ```
pub fn from_python_graph(pygraph: &Bound<'_, PyAny>) -> Graph {
    let mut graph = Graph::new();

    let nodes = get_nodes(pygraph).expect("Failed to extract nodes from Python graph");
    let edges = get_edges(pygraph).expect("Failed to extract edges from Python graph");

    // Add all nodes first
    for node in nodes {
        graph.nodes.insert(node);
        graph.adjacency_list.entry(node).or_default();
    }

    // Add all edges
    for (from, to) in edges {
        graph.add_edge(from, to);
    }

    graph.finalize();
    graph
}

/// Validates that a Python object appears to be a supported graph
///
/// Performs basic checks to determine if the object has the expected
/// methods for NetworkX or igraph graphs.
///
/// # Arguments
/// * `obj` - Python object to validate
///
/// # Returns
/// True if the object appears to be a supported graph format
pub fn is_supported_graph(obj: &Bound<'_, PyAny>) -> bool {
    // Check for NetworkX-style graph
    if obj.hasattr("nodes").unwrap_or(false) && obj.hasattr("edges").unwrap_or(false) {
        return true;
    }

    // Check for igraph-style graph
    if obj.hasattr("vs").unwrap_or(false) && obj.hasattr("get_edgelist").unwrap_or(false) {
        return true;
    }

    false
}

/// Gets basic statistics from a Python graph without full conversion
///
/// Useful for debugging and validation before expensive operations.
///
/// # Arguments
/// * `pygraph` - Python graph object
///
/// # Returns
/// Tuple of (number_of_nodes, number_of_edges)
pub fn get_graph_stats(pygraph: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let nodes = get_nodes(pygraph)?;
    let edges = get_edges(pygraph)?;
    Ok((nodes.len(), edges.len()))
}
