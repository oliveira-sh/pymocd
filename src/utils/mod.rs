//! utils/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::{debug, graph::*};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::{BTreeMap, HashMap};

pub fn normalize_community_ids(
    graph: &Graph,
    partition: Partition,
) -> Partition {
    let mut new_partition: BTreeMap<NodeId, CommunityId> = BTreeMap::new();
    let mut id_mapping: HashMap<CommunityId, CommunityId> = HashMap::new();
    let mut next_id: CommunityId = 0;

    for &node in graph.nodes.iter() {
        let is_isolated = match graph.adjacency_list.get(&node) {
            Some(neighbors) => neighbors.is_empty(),
            None => true, // if hasnt adjacency_list, it is isolated
        };

        if is_isolated {
            new_partition.insert(node, -1);
        } else {
            match partition.get(&node) {
                Some(&orig_comm) if orig_comm != -1 => {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        id_mapping.entry(orig_comm)
                    {
                        e.insert(next_id);
                        next_id += 1;
                    }
                    let mapped = *id_mapping.get(&orig_comm).unwrap();
                    new_partition.insert(node, mapped);
                }
                _ => {
                    new_partition.insert(node, -1);
                }
            }
        }
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
pub fn get_nodes(graph: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
    // Try NetworkX: graph.nodes()
    if let Ok(nx_nodes) = graph.call_method0("nodes") {
        let iter = nx_nodes.call_method0("__iter__")?;
        let mut nodes: Vec<NodeId> = Vec::new();

        for node_obj in iter.try_iter()? {
            let node_id: NodeId = node_obj?.extract()?;
            nodes.push(node_id);
        }
        return Ok(nodes);
    }

    // if fail, try igraph: graph.vs
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
        "Não foi possível obter a lista de nós de NetworkX nem de igraph",
    ))
}
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
pub fn build_graph(nodes: Vec<NodeId>, edges: Vec<(NodeId, NodeId)>) -> Graph {
    let mut graph = Graph::new();
    
    for node in nodes {
        graph.nodes.insert(node);
        graph.adjacency_list.entry(node).or_default();
    }
    
    for (from, to) in edges {
        graph.edges.push((from, to));
        
        graph.nodes.insert(from);
        graph.nodes.insert(to);

        graph.adjacency_list.entry(from).or_default().push(to);
        graph.adjacency_list.entry(to).or_default().push(from);
    }
    
    graph
}

