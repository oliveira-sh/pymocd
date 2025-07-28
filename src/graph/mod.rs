//! graph/mods.rs - Optimized graph implementation for community detection
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::debug;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub type NodeId = i32;
pub type CommunityId = i32;
pub type Partition = FxHashMap<NodeId, CommunityId>;

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

pub fn get_nodes(graph: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
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

pub fn get_edges(graph: &Bound<'_, PyAny>) -> PyResult<Vec<(NodeId, NodeId)>> {
    let edges_iter = match graph.call_method0("edges") {
        Ok(nx_edges) => nx_edges.call_method0("__iter__")?,
        Err(_) => {
            debug!(warn, "networkx.Graph() not found, trying igraph.Graph()");
            match graph.call_method0("get_edgelist") {
                Ok(ig_edges) => ig_edges.call_method0("__iter__")?,
                Err(_) => {
                    debug!(err, "supported graph libraries not found");
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

impl Graph {
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

    pub fn from_python(pygraph: &Bound<'_, PyAny>) -> Self {
        let mut graph = Graph::new();
        let nodes = get_nodes(pygraph).unwrap();
        let edges = get_edges(pygraph).unwrap();

        for node in nodes {
            graph.nodes.insert(node);
            graph.adjacency_list.entry(node).or_default();
        }

        for (from, to) in edges {
            if from == to {
                continue; // Skip self-loops
            }
            
            graph.edges.push((from, to));
            graph.nodes.insert(from);
            graph.nodes.insert(to);
            
            let edge_key = if from < to { (from, to) } else { (to, from) };
            graph.edge_lookup.insert(edge_key);
            
            graph.adjacency_list.entry(from).or_default().push(to);
            graph.adjacency_list.entry(to).or_default().push(from);
        }

        graph.max_degree = 0;
        graph.total_degree = 0;
        
        for (node, neighbors) in &graph.adjacency_list {
            let degree = neighbors.len();
            graph.degrees.insert(*node, degree);
            graph.max_degree = graph.max_degree.max(degree);
            graph.total_degree += degree;
        }

        graph.finalize();
        graph
    }

    pub fn print(&self) {
        debug!(
            debug,
            "G = ({},{}) | Max Degree: {} | Avg Degree: {:.2}",
            self.num_nodes(),
            self.num_edges(),
            self.max_degree,
            self.total_degree as f64 / self.num_nodes() as f64
        );
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        if from == to {
            return;
        }

        let edge_key = if from < to { (from, to) } else { (to, from) };
        if self.edge_lookup.contains(&edge_key) {
            return;
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

    pub fn finalize(&mut self) {
        self.node_vec = self.nodes.iter().copied().collect();
        self.node_vec.sort_unstable();

        self.degrees.shrink_to_fit();
        self.edge_lookup.shrink_to_fit();

        for neighbors in self.adjacency_list.values_mut() {
            neighbors.sort_unstable();
            neighbors.shrink_to_fit();
        }
    }

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

    #[inline(always)]
    pub fn neighbors(&self, node: &NodeId) -> &[NodeId] {
        self.adjacency_list.get(node).map_or(&[], |x| x)
    }

    #[inline(always)]
    pub fn degree(&self, node: &NodeId) -> usize {
        *self.degrees.get(node).unwrap_or(&0)
    }

    #[inline(always)]
    pub fn has_edge(&self, from: NodeId, to: NodeId) -> bool {
        let edge_key = if from < to { (from, to) } else { (to, from) };
        self.edge_lookup.contains(&edge_key)
    }

    #[inline(always)]
    pub fn nodes_iter(&self) -> impl Iterator<Item = &NodeId> {
        self.node_vec.iter()
    }

    #[inline(always)]
    pub fn nodes_vec(&self) -> &Vec<NodeId> {
        &self.node_vec
    }

    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline(always)]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    #[inline(always)]
    pub fn precompute_degrees(&self) -> &FxHashMap<NodeId, usize> {
        &self.degrees
    }

    #[inline(always)]
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline(always)]
    pub fn total_degree(&self) -> usize {
        self.total_degree
    }

    #[inline(always)]
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.total_degree as f64 / self.num_nodes() as f64
        }
    }

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

pub struct GraphMemoryStats {
    pub nodes_memory: usize,
    pub edges_memory: usize,
    pub adjacency_memory: usize,
    pub degrees_memory: usize,
    pub edge_lookup_memory: usize,
}

impl GraphMemoryStats {
    pub fn total(&self) -> usize {
        self.nodes_memory
            + self.edges_memory
            + self.adjacency_memory
            + self.degrees_memory
            + self.edge_lookup_memory
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_graph_num_nodes() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.max_degree(), 3);
        assert_eq!(graph.total_degree(), 6);
    }

    #[test]
    fn test_neighbors() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        let mut neighbors: Vec<NodeId> = graph.neighbors(&0).to_vec();
        neighbors.sort();
        assert_eq!(neighbors, [1, 2, 4]);
    }

    #[test]
    fn test_degree_access() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);
        graph.finalize();

        assert_eq!(graph.degree(&0), 3);
        assert_eq!(graph.degree(&1), 1);
        assert_eq!(graph.degree(&2), 1);
        assert_eq!(graph.degree(&4), 1);
    }

    #[test]
    fn test_has_edge() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.finalize();

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0));
        assert!(graph.has_edge(0, 2));
        assert!(!graph.has_edge(1, 2));
    }

    #[test]
    fn test_duplicate_edge_prevention() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 0);
        graph.add_edge(0, 1);
        graph.finalize();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.degree(&0), 1);
        assert_eq!(graph.degree(&1), 1);
    }

    #[test]
    fn test_nodes_iteration() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);
        graph.finalize();

        let mut nodes: Vec<NodeId> = graph.nodes_iter().copied().collect();
        nodes.sort();
        assert_eq!(nodes, [0, 1, 2, 3]);
    }
}
