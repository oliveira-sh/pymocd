//! /objective.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{Graph, Partition};
use rustc_hash::FxHashMap as HashMap;

pub fn calculate_objectives(graph: &Graph, partition: &Partition) -> [f64; 2] {
    let total_edges = graph.num_edges() as f64;
    if total_edges == 0.0 {
        return [0.0, 0.0];
    }

    let mut community_degrees: HashMap<i32, f64> = HashMap::default();
    let mut intra_edges = 0.0;

    // Use pre-computed degrees instead of parameter
    let degrees = graph.precompute_degrees();

    // Single pass through all nodes using faster iteration
    for &node in graph.nodes_iter() {
        if let Some(&comm) = partition.get(&node) {
            let degree = *degrees.get(&node).unwrap_or(&0) as f64;
            *community_degrees.entry(comm).or_default() += degree;
        }
    }

    // Use edges vector directly - it's already optimized
    for &(u, v) in &graph.edges {
        if let (Some(&comm_u), Some(&comm_v)) = (partition.get(&u), partition.get(&v)) {
            if comm_u == comm_v {
                intra_edges += 1.0;
            }
        }
    }

    let total_edges_doubled = 2.0 * total_edges;
    let inter: f64 = community_degrees
        .values()
        .map(|&degree| (degree / total_edges_doubled).powi(2))
        .sum();

    let intra = 1.0 - (intra_edges / total_edges);
    [intra, inter]
}

