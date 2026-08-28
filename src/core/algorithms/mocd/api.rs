//! The `Mocd` pyclass: the two entry points (MOCD-Q front, MOCD-D selection).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{pyclass, pymethods};

use crate::core::graph::{Graph, Partition, normalize_community_ids};

use super::defaults::{
    DEFAULT_CROSS_RATE, DEFAULT_DEBUG_LEVEL, DEFAULT_MUT_RATE, DEFAULT_NUM_GENS, DEFAULT_POP_SIZE,
    DEFAULT_RAND_NETWORKS, EPSIZE_CAP,
};
use super::null_model::generate_random_networks;
use super::pesa2::{Solution, evolutionary_phase};
use super::selection::min_max_selection;

#[pyclass]
pub struct Mocd {
    graph: Graph,
    debug_level: i8,
    rand_networks: usize,
    pop_size: usize,
    ep_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
}

// `labels` is positional: slot `i` is `graph.nodes_vec()[i]`, the same order
// `NodeIndex::build` assigns dense indices in.
fn labels_to_partition(graph: &Graph, labels: &[i32]) -> Partition {
    graph
        .nodes_vec()
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, labels[i]))
        .collect()
}

impl Mocd {
    pub fn envolve(&self) -> Vec<Solution> {
        if self.debug_level >= 1 {
            self.graph.print();
        }

        evolutionary_phase(
            &self.graph,
            self.debug_level,
            self.num_gens,
            self.pop_size,
            self.ep_size,
            self.cross_rate,
            self.mut_rate,
        )
    }
}

#[pymethods]
impl Mocd {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = DEFAULT_DEBUG_LEVEL,
        rand_networks = DEFAULT_RAND_NETWORKS,
        pop_size = DEFAULT_POP_SIZE,
        num_gens = DEFAULT_NUM_GENS,
        cross_rate = DEFAULT_CROSS_RATE,
        mut_rate = DEFAULT_MUT_RATE
    ))]
    pub fn new(
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        rand_networks: usize,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
    ) -> PyResult<Self> {
        let graph = Graph::from_python(graph);

        Ok(Self {
            graph,
            debug_level,
            rand_networks,
            pop_size,
            ep_size: pop_size.clamp(1, EPSIZE_CAP),
            num_gens,
            cross_rate,
            mut_rate,
        })
    }

    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let first_front = self.envolve();

        Ok(first_front
            .into_iter()
            .map(|ind| {
                (
                    normalize_community_ids(
                        &self.graph,
                        labels_to_partition(&self.graph, &ind.labels),
                    ),
                    ind.objectives,
                )
            })
            .collect())
    }

    pub fn run(&self) -> PyResult<Partition> {
        let archive = self.envolve();

        let best_solution = {
            let random_networks = generate_random_networks(&self.graph, self.rand_networks);

            let random_archives: Vec<Vec<Solution>> = random_networks
                .iter()
                .map(|random_graph| {
                    // Control fronts need the FULL budget (Shi 2012, §3.2): an
                    // under-evolved random front misses the fragmented region
                    // and makes the real fragmented extreme spuriously deviant.
                    evolutionary_phase(
                        random_graph,
                        self.debug_level,
                        self.num_gens,
                        self.pop_size,
                        self.ep_size,
                        self.cross_rate,
                        self.mut_rate,
                    )
                })
                .collect();
            min_max_selection(&archive, &random_archives)
        };

        Ok(normalize_community_ids(
            &self.graph,
            labels_to_partition(&self.graph, &best_solution.labels),
        ))
    }
}
