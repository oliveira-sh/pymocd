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

    pub fn get_node_mapping(&self) -> (Vec<NodeId>, HashMap<NodeId, usize>) {
        let nodes_vec: Vec<NodeId> = self.nodes.iter().cloned().collect();
        let node_to_idx: HashMap<NodeId, usize> = nodes_vec
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx))
            .collect();
        (nodes_vec, node_to_idx)
    }

    /// Precomputes the degree of each node.
    pub fn precompute_degrees(&self) -> HashMap<NodeId, usize> {
        let mut degrees = HashMap::default();
        for &node in &self.nodes {
            degrees.insert(node, self.adjacency_list[&node].len());
        }
        degrees
    }

    /// Detects key nodes for community detection based on degree centrality.
    /// 
    /// This function implements an algorithm that:
    /// 1. Calculates the average degree of all nodes
    /// 2. Identifies nodes with above-average degree
    /// 3. Iteratively selects nodes with maximum degree, excluding previously selected nodes' neighbors
    /// 
    /// # Arguments
    /// * `&self` - Reference to the Graph instance
    /// 
    /// # Returns
    /// * `HashSet<NodeId>` - Set of key nodes identified for community detection
    pub fn detect_key_nodes(&self) -> HashSet<NodeId> {
        let mut kv = HashSet::default();
        let avg_degree = self.nodes.iter()
            .map(|&node| self.adjacency_list[&node].len())
            .sum::<usize>() as f64 / self.nodes.len() as f64;
        let mut lset: HashSet<NodeId> = self.nodes.iter()
            .filter(|&&node| self.adjacency_list[&node].len() as f64 > avg_degree)
            .cloned()
            .collect();

        while !lset.is_empty() {
            let kv_i = *lset.iter()
                .max_by_key(|&&node| self.adjacency_list[&node].len())
                .unwrap();
            kv.insert(kv_i);
            let neighbors = self.adjacency_list[&kv_i].clone();
            lset.remove(&kv_i);
            for neighbor in neighbors {
                lset.remove(&neighbor);
            }
        }
        kv
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

    #[test]
    fn test_detect_key_nodes() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(4, 5);
        graph.add_edge(4, 6);
        graph.add_edge(4, 7);
        
        let key_nodes = graph.detect_key_nodes();
        
        // Verify results: should contain central nodes of both stars
        assert_eq!(key_nodes.len(), 2, "Should detect exactly 2 key nodes");
        assert!(key_nodes.contains(&0), "Node 0 should be a key node");
        assert!(key_nodes.contains(&4), "Node 4 should be a key node");
    }

    #[test]
    fn test_detect_key_nodes_empty() {     
        // Cycle graph where all degrees equal average
        let mut graph2 = Graph::new();
        graph2.add_edge(0, 1);
        graph2.add_edge(1, 2);
        graph2.add_edge(2, 3);
        graph2.add_edge(3, 0);
        
        let key_nodes2 = graph2.detect_key_nodes();        
        assert!(key_nodes2.is_empty(), "Key nodes should be empty for cycle graph");
    }
}
