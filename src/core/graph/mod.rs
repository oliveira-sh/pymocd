//! graph representations for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

pub type NodeId = i32;
pub type CommunityId = i32;
pub type Partition = FxHashMap<NodeId, CommunityId>;

mod adj;
mod csr;
#[cfg(test)]
pub mod karate;

pub use adj::{Graph, get_edges, get_nodes};
pub use csr::CsrGraph;

/// Relabel communities to dense ids starting at 0.
///
/// Every node of `graph` appears in the result. A degree-0 node is forced to
/// `-1` even when `partition` assigns it a community: isolated nodes are not
/// members of anything, and downstream scoring relies on that marker. Nodes
/// missing from `partition`, or already carrying `-1`, also map to `-1`.
pub fn normalize_community_ids(graph: &Graph, partition: Partition) -> Partition {
    let mut new_partition: FxHashMap<NodeId, CommunityId> = FxHashMap::default();
    let mut id_mapping: FxHashMap<CommunityId, CommunityId> = FxHashMap::default();
    let mut next_id: CommunityId = 0;

    for &node in &graph.nodes {
        let is_isolated = graph.adjacency_list.get(&node).is_none_or(Vec::is_empty);

        if is_isolated {
            new_partition.insert(node, -1);
        } else {
            match partition.get(&node) {
                Some(&orig_comm) if orig_comm != -1 => {
                    let mapped = *id_mapping.entry(orig_comm).or_insert_with(|| {
                        let dense = next_id;
                        next_id += 1;
                        dense
                    });
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

/// Read a Python `{node: community}` dict into a [`Partition`].
pub fn to_partition(py_dict: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut part: FxHashMap<i32, i32> = FxHashMap::default();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}
