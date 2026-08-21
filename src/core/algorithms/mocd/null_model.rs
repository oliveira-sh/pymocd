//! The Erdős–Rényi control networks MOCD-D scores its front against.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{RngExt, rng};
use std::collections::HashSet;

use crate::core::graph::{Graph, NodeId};

/// `num_networks` Erdős–Rényi `G(n, m)` graphs with the same node and edge
/// counts as `original` ("random networks with the same scale", Shi 2012 §3.2).
pub fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    let nodes: Vec<NodeId> = original.nodes_vec().clone();
    let n = nodes.len();
    let m = original.edges.len();
    (0..num_networks)
        .map(|_| {
            let mut r = rng();
            let mut present: HashSet<(NodeId, NodeId)> = HashSet::with_capacity(m);
            // Every node is pre-inserted, edges or not, so `n` matches
            // `original`; the rest goes through `add_edge`/`finalize` so the
            // derived fields (node_vec, degrees, edge_lookup) get built.
            let mut random_graph = Graph::new();
            for &node in &nodes {
                random_graph.nodes.insert(node);
                random_graph.adjacency_list.entry(node).or_default();
            }
            while present.len() < m {
                let a = nodes[r.random_range(0..n)];
                let b = nodes[r.random_range(0..n)];
                if a == b {
                    continue;
                }
                let key = if a <= b { (a, b) } else { (b, a) };
                if present.insert(key) {
                    random_graph.add_edge(key.0, key.1);
                }
            }
            random_graph.finalize();
            random_graph
        })
        .collect()
}
