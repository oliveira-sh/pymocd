//! utils/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::{debug, graph::*};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::{BTreeMap, HashMap};

pub fn normalize_community_ids(partition: Partition) -> Partition {
    let mut new_partition: BTreeMap<i32, i32> = Partition::new();
    let mut id_mapping: HashMap<i32, i32> = HashMap::new();
    let mut next_id: i32 = 0;

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
pub fn to_partition(py_dict: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut part: BTreeMap<i32, i32> = BTreeMap::new();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}
/// Try NetworkX's `edges()` first. If that fails, fall back to igraph's `get_edgelist()`.
pub fn get_edges(graph: &Bound<'_, PyAny>) -> PyResult<Vec<(NodeId, NodeId)>> {
    let edges_iter = match graph.call_method0("edges") {
        Ok(nx_edges) => {
            // NetworkX: `edges()` returns an EdgeView; get its iterator
            nx_edges.call_method0("__iter__")?
        }
        Err(_) => {
            debug!(warn, "networkx.Graph() not found, trying igraph.Graph()");
            match graph.call_method0("get_edgelist") {
                Ok(ig_edges) => {
                    // igraph: `get_edgelist()` returns a list of edge tuples; get its iterator
                    ig_edges.call_method0("__iter__")?
                }
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
pub fn build_graph(edges: Vec<(NodeId, NodeId)>) -> Graph {
    let mut graph: Graph = Graph::new();
    for (from, to) in edges {
        graph.add_edge(from, to);
    }
    graph
}
