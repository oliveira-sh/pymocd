//! utils/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

pub fn normalize_community_ids(graph: &Graph, partition: Partition) -> Partition {
    let mut new_partition: FxHashMap<NodeId, CommunityId> = FxHashMap::default();
    let mut id_mapping: FxHashMap<CommunityId, CommunityId> = FxHashMap::default();
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
    let mut part: FxHashMap<i32, i32> = FxHashMap::default();
    for (node, comm) in py_dict.iter() {
        part.insert(node.extract::<NodeId>()?, comm.extract::<CommunityId>()?);
    }
    Ok(part)
}
