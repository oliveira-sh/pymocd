//! utils/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::*;

use std::collections::{BTreeMap, HashMap};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub fn normalize_community_ids(partition: Partition) -> Partition {
    let mut new_partition = Partition::new();
    let mut id_mapping = HashMap::new();
    let mut next_id = 0;

    // Create a new mapping for community IDs
    for (node_id, &community_id) in partition.iter() {
        if let std::collections::hash_map::Entry::Vacant(e) = id_mapping.entry(community_id) {
            e.insert(next_id);
            next_id += 1;
        }
        new_partition.insert(*node_id, *id_mapping.get(&community_id).unwrap());
    }

    new_partition
}

/// Convert Python dict to Rust partition
#[allow(dead_code)]
pub fn to_partition(py_dict: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut part = BTreeMap::new();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}

/// Get edges from NetworkX graph
pub fn get_edges(graph: &Bound<'_, PyAny>) -> PyResult<Vec<(NodeId, NodeId)>> {
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
pub fn build_graph(edges: Vec<(NodeId, NodeId)>) -> Graph {
    let mut graph = Graph::new();
    for (from, to) in edges {
        graph.add_edge(from, to);
    }
    graph
}
