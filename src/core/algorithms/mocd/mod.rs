//! Shi-MOCD (Shi, Yan, Cai, Wu 2012): a self-contained, single-threaded
//! PESA-II (Corne et al. 2001) community detector over Shi's locus-based
//! adjacency genome (`locus.rs`, Park & Song scheme, §3.1.3) and
//! decomposed-modularity objectives, with both model selectors — MOCD-Q (max
//! modularity, Eq. 3.8) and MOCD-D (max-min distance to random-network
//! control fronts, Eqs. 3.9–3.11). Deliberately avoids the shared PESA-II
//! engine and Rayon so cost tracks the paper; hot-path state is dense
//! Vec-indexed, with the shared `Partition` built only at this boundary.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod defaults;
mod locus;
mod model_selection;
mod objectives;
mod pesa2;
pub use defaults::*;

use crate::core::graph::{Graph, Partition};
use pesa2::{Solution, evolutionary_phase};

use pyo3::{pyclass, pymethods};

use crate::core::graph::normalize_community_ids;

use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub struct Mocd {
    graph: Graph,
    debug_level: i8,
    rand_networks: usize,
    pop_size: usize,
    // EP capacity; not Python-visible, defaults to `min(pop_size, EPSIZE_CAP)`.
    ep_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
}

/// Boundary conversion: dense labels (positions follow `graph.nodes_vec()`
/// order, the same order `NodeIndex::build` uses) -> shared `Partition`.
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

        Ok(Mocd {
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
            let random_networks =
                model_selection::generate_random_networks(&self.graph, self.rand_networks);

            let random_archives: Vec<Vec<Solution>> = random_networks
                .iter()
                .map(|random_graph| {
                    // Control fronts need the FULL budget (Shi 2012, §3.2): an
                    // under-evolved random front misses the fragmented region,
                    // making the real fragmented extreme spuriously "most
                    // deviant" instead of the genuine community bulge.
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
            model_selection::min_max_selection(&archive, &random_archives)
        };

        Ok(normalize_community_ids(
            &self.graph,
            labels_to_partition(&self.graph, &best_solution.labels),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::pesa2::evolutionary_phase;
    use crate::core::graph::Graph;
    use std::collections::HashSet;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn shi_mocd_max_q_is_two_community_split() {
        let g = two_triangles();
        let archive = evolutionary_phase(&g, 0, 100, 100, 100, 0.6, 0.4);
        assert!(!archive.is_empty(), "empty PESA-II archive");
        // MOCD-Q (Shi Eq. 3.8): argmin(intra + inter) = argmax Q.
        let best = archive
            .iter()
            .min_by(|a, b| {
                (a.objectives[0] + a.objectives[1])
                    .partial_cmp(&(b.objectives[0] + b.objectives[1]))
                    .unwrap()
            })
            .unwrap();
        // Q = 1 - intra - inter (Shi Eq. 3.7); objectives = [intra, inter].
        let q = 1.0 - best.objectives[0] - best.objectives[1];
        assert!(q > 0.0, "Q = {q}");
        // Dense labels follow g.nodes_vec() (sorted), so position == node id here.
        assert_ne!(best.labels[0], best.labels[3], "triangles not split");
        let comms: HashSet<i32> = best.labels.iter().copied().collect();
        assert_eq!(comms.len(), 2, "communities = {comms:?}");
    }
}
