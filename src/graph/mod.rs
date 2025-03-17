//! operators/mod.rs
//! Optimized Graph definitions
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use std::collections::BTreeMap;

pub type NodeId = i32;
pub type CommunityId = i32;
pub type Partition = BTreeMap<NodeId, CommunityId>;

#[derive(Debug, Clone)]
pub struct Graph {
    pub edges: Vec<(NodeId, NodeId)>,
    pub nodes: HashSet<NodeId>,
    pub adjacency_list: HashMap<NodeId, Vec<NodeId>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            edges: Vec::new(),
            nodes: HashSet::default(),
            adjacency_list: HashMap::default(),
        }
    }

    pub fn print(&self) {
        println!(
            "[graph/mod.rs]: graph n/e: {}/{}",
            self.num_nodes(),
            self.num_edges(),
        );
    }
    
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.push((from, to));
        self.nodes.insert(from);
        self.nodes.insert(to);

        // Update adjacency list
        self.adjacency_list.entry(from).or_default().push(to);
        self.adjacency_list.entry(to).or_default().push(from);
    }

    pub fn neighbors(&self, node: &NodeId) -> &[NodeId] {
        self.adjacency_list.get(node).map_or(&[], |x| x)
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Precomputes the degree of each node.
    pub fn precompute_degrees(&self) -> HashMap<NodeId, usize> {
        let mut degrees = HashMap::default();
        for &node in &self.nodes {
            degrees.insert(node, self.adjacency_list[&node].len());
        }
        degrees
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

        assert_eq!(graph.num_nodes(), 4);
    }

    #[test]
    fn test_neighbors() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);

        assert_eq!(graph.neighbors(&0), [1, 2, 4]);
    }

    #[test]
    fn test_precompute_degrees() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);

        let mut expected = HashMap::default();
        expected.insert(0, 3);
        expected.insert(2, 1);
        expected.insert(4, 1);
        expected.insert(1, 1);

        assert_eq!(graph.precompute_degrees(), expected);
    }

    #[test]
    fn test_graph_num_edges() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);

        assert_eq!(graph.num_edges(), 3);
    }
}
