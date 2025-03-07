//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::BTreeMap;

mod algorithms;
mod graph;
pub mod operators;
mod utils;

use graph::{CommunityId, Graph, NodeId, Partition};
use utils::args::AGArgs as AlgorithmConfig;

// ================================================================================================
// Py functions
// ================================================================================================

/// Takes a NetworkX Graph as input and performs community detection
///
/// # Parameters
/// - `graph` (networkx.Graph): The graph on which to perform community detection
/// - `debug` (i8, optional): Enable debug output. Large the num, large the debug. [d=0,1,2,3]
///
/// # Returns
/// - dict[int, int]: Mapping of node IDs to their detected community IDs
#[pyfunction(name = "pesa_ii_minimax")]
#[pyo3(signature = (graph, debug = 0))]
fn pesa_ii_minimax(py: Python<'_>, graph: &Bound<'_, PyAny>, debug: i8) -> PyResult<BTreeMap<i32, i32>> {
    let edges = get_edges(graph)?;
    let config = AlgorithmConfig::lib_args(debug);

    if config.debug >= 2 {
        println!("{:?}", config);
    }

    py.allow_threads(|| {
        let graph = build_graph(edges);
        let (communities, _, _) = algorithms::pesa_ii(&graph, config, false);

        Ok(communities)
    })
}

/// Takes a NetworkX Graph as input and performs community detection
///
/// # Parameters
/// - `graph` (networkx.Graph): The graph on which to perform community detection
/// - `debug` (i8, optional): Enable debug output. Large the num, large the debug. [d=0,1,2,3]
///
/// # Returns
/// - dict[int, int]: Mapping of node IDs to their detected community IDs
#[pyfunction(name = "pesa_ii_maxq")]
#[pyo3(signature = (graph, debug = 0))]
fn pesa_ii_maxq(py: Python<'_>, graph: &Bound<'_, PyAny>, debug: i8) -> PyResult<BTreeMap<i32, i32>> {
    let edges = get_edges(graph)?;
    let config = AlgorithmConfig::lib_args(debug);

    if config.debug >= 2 {
        println!("{:?}", config);
    }

    py.allow_threads(|| {
        let graph = build_graph(edges);
        let (communities, _, _) = algorithms::pesa_ii(&graph, config, true);

        Ok(communities)
    })
}

/// Takes a NetworkX Graph as input and performs community detection
///
/// # Parameters
/// - `graph` (networkx.Graph): The graph on which to perform community detection
/// - `debug` (i8, optional): Enable debug output. Large the num, large the debug. [d=0,1,2,3]
///
/// # Returns
/// - dict[int, int]: Mapping of node IDs to their detected community IDs
#[pyfunction(name = "nsga_ii")]
#[pyo3(signature = (graph, debug = 0))]
fn nsga_ii(py: Python<'_>, graph: &Bound<'_, PyAny>, debug: i8) -> PyResult<BTreeMap<i32, i32>> {
    let edges = get_edges(graph)?;
    let config = AlgorithmConfig::lib_args(debug);

    if config.debug >= 2 {
        println!("{:?}", config);
    }

    py.allow_threads(|| {
        let graph = build_graph(edges);
        let (communities, _, _) = algorithms::nsga_ii(&graph, config);

        Ok(communities)
    })
}

/// Calculates the modularity score for a given graph and community partition
///
/// # Parameters
/// - `graph` (networkx.Graph): The graph to analyze
/// - `partition` (dict[int, int]): Dictionary mapping nodes to community IDs
///
/// # Returns
/// - float: Modularity score based on (Shi, 2012) multi-objective modularity equation
#[pyfunction(name = "modularity")]
fn fitness(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let edges = get_edges(graph)?;
    let graph = build_graph(edges);

    Ok(operators::get_modularity_from_partition(
        &to_partition(partition)?,
        &graph,
    ))
}

// ================================================================================================
// Helper functions
// ================================================================================================

/// Convert Python dict to Rust partition
fn to_partition(py_dict: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut part = BTreeMap::new();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}

/// Get edges from NetworkX graph
fn get_edges(graph: &Bound<'_, PyAny>) -> PyResult<Vec<(NodeId, NodeId)>> {
    let mut edges = Vec::new();
    let edges_iter = graph.call_method0("edges")?.call_method0("__iter__")?;

    for edge in edges_iter.try_iter()? {
        let edge = edge?;
        let from = edge.get_item(0)?.extract()?;
        let to = edge.get_item(1)?.extract()?;
        edges.push((from, to));
    }

    Ok(edges)
}

/// Build Graph from edges
fn build_graph(edges: Vec<(NodeId, NodeId)>) -> Graph {
    let mut graph = Graph::new();
    for (from, to) in edges {
        graph.add_edge(from, to);
    }
    graph
}

// ================================================================================================
// Module
// ================================================================================================

#[pymodule]
fn re_mocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pesa_ii_maxq, m)?)?;
    m.add_function(wrap_pyfunction!(pesa_ii_minimax, m)?)?;

    m.add_function(wrap_pyfunction!(nsga_ii, m)?)?;
    m.add_function(wrap_pyfunction!(fitness, m)?)?;
    Ok(())
}

/* ================================================================================================
// Deprecated
// ================================================================================================

/// Performs community detection on a graph from an edge list file
#[pyfunction(name = "from_file")]
#[pyo3(signature = (file_path))]
fn from_file(file_path: String) -> PyResult<BTreeMap<i32, i32>> {
    let config = AlgorithmConfig::parse(&vec!["--library-".to_string(), file_path]);
    if config.debug {
        println!("[Detection]: Config: {:?}", config);
    }

    let graph = Graph::from_edgelist(Path::new(&config.file_path))?;
    let (communities, _, _) = algorithms::pesa_ii(&graph, config);

    Ok(communities)
}


*/