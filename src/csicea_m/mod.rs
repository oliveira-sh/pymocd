//! Community Structure Identification Parallel Co‑evolutionary Algorithm (CSICEA)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

// ================================================================================================

mod moea;
use moea::Moea;

use crate::graph::{Graph, NodeId};
use crate::utils;
use ndarray::Array2;
use pyo3::{prelude::*, types::PyAny};
use rustc_hash::FxHashMap;

// ================================================================================================

#[pyfunction]
#[pyo3(signature = (graph,
        popsize = 100,
        t = 40,
        max_gen = 50,
        cross_rate = 0.8,
        mutate_rate = 0.8,
        alpha = 0.4,
        verbose = false
    ))]
pub fn csicea(
    graph: &Bound<'_, PyAny>,
    popsize: usize,
    t: usize,
    max_gen: usize,
    cross_rate: f64,
    mutate_rate: f64,
    alpha: f64,
    verbose: bool,
) -> PyResult<FxHashMap<usize, usize>> {
    let nodes: Result<Vec<i32>, PyErr> = utils::get_nodes(graph);
    let edges: Result<Vec<(i32, i32)>, PyErr> = utils::get_edges(graph);
    let graph: Graph = utils::build_graph(nodes.unwrap(), edges.unwrap());

    let node_count: usize = graph.num_nodes();
    if node_count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Unable to get node list from NetworkX or igraph",
        ));
    }

    let mut node_id_to_index: FxHashMap<NodeId, usize> =
        FxHashMap::with_capacity_and_hasher(node_count, Default::default());
    let mut index_to_node_id: Vec<NodeId> = Vec::with_capacity(node_count);

    for (i, &node_id) in graph.nodes.iter().enumerate() {
        node_id_to_index.insert(node_id, i);
        index_to_node_id.push(node_id);
    }

    let mut adj_matrix = Array2::<f64>::zeros((node_count, node_count));

    for chunk in graph.edges.chunks(1000) {
        for &(from, to) in chunk {
            if let (Some(&i), Some(&j)) = (node_id_to_index.get(&from), node_id_to_index.get(&to)) {
                adj_matrix[[i, j]] = 1.0;
                adj_matrix[[j, i]] = 1.0; // TODO: undirected, maybe directed later?
            }
        }
    }

    let mut moea = Moea::new(
        adj_matrix,
        popsize,
        t,
        max_gen,
        cross_rate,
        mutate_rate,
        alpha,
        verbose,
    );

    let result_by_index = moea.run();

    // partition = NodeId -> CommunityId
    let mut partition =
        FxHashMap::with_capacity_and_hasher(result_by_index.len(), Default::default());
    for (idx, comm) in result_by_index {
        if idx < index_to_node_id.len() {
            let node_id = index_to_node_id[idx];
            partition.insert(node_id as usize, comm);
        }
    }

    Ok(partition)
}
