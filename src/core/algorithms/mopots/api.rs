//! MO-POTS pyclass: parameter checks, the NSGA-II run and its three front readers.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core::graph::normalize_community_ids;
use crate::core::graph::{Graph, Partition};

use super::defaults::{
    DEFAULT_CROSS_RATE, DEFAULT_DEBUG_LEVEL, DEFAULT_MUT_RATE, DEFAULT_NUM_GENS, DEFAULT_POP_SIZE,
};
use super::front::{max_q_selection, resolution_ladder};
use super::nsga2::{Individual, TOURNAMENT_SIZE, evolve};
use super::objectives;

fn is_probability(rate: f64) -> bool {
    rate.is_finite() && (0.0..=1.0).contains(&rate)
}

/// NSGA-II over the exact `(cut, pair)` split of the Constant Potts Model; the
/// method, its objectives and its divergences are in this module's README.
///
/// A DiGraph's reciprocal arcs are kept as two edges, which inflates `m` and
/// therefore every reported `gamma`. Isolated nodes are outside both objectives
/// and come back as community `-1`.
#[pyclass]
pub struct MoPots {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
}

fn search(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    debug_level: i8,
) -> Vec<Individual> {
    let individuals = evolve(
        graph,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        TOURNAMENT_SIZE,
        |inds: &mut [Individual]| {
            // (cut, pair) is pure Rust, so evaluation never needs the GIL
            inds.par_iter_mut().for_each(|ind| {
                let metrics = objectives::calculate_objectives(graph, &ind.partition, true);
                ind.objectives = vec![metrics.cut, metrics.pair];
            });
        },
        |generation, num_gens, pop: &[Individual]| {
            if debug_level >= 1 && (generation % 10 == 0 || generation == num_gens - 1) {
                let first_front_size = pop.iter().filter(|ind| ind.rank == 1).count();

                debug!(
                    debug,
                    "NSGA-II: Gen {} | 1st Front/Pop: {}/{}",
                    generation,
                    first_front_size,
                    pop.len()
                );
            }
        },
    );

    individuals
        .into_iter()
        .filter(|ind| ind.rank == 1)
        .collect()
}

impl MoPots {
    fn rank_one_front(&self) -> Vec<Individual> {
        search(
            &self.graph,
            self.pop_size,
            self.num_gens,
            self.cross_rate,
            self.mut_rate,
            self.debug_level,
        )
    }

    fn is_empty(&self) -> bool {
        self.graph.num_nodes() == 0
    }
}

#[pymethods]
impl MoPots {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = DEFAULT_DEBUG_LEVEL,
        pop_size = DEFAULT_POP_SIZE,
        num_gens = DEFAULT_NUM_GENS,
        cross_rate = DEFAULT_CROSS_RATE,
        mut_rate = DEFAULT_MUT_RATE
    ))]
    pub fn new(
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
    ) -> PyResult<Self> {
        // an unchecked parameter panics deep inside the search, some of it in a rayon worker
        if pop_size == 0 {
            return Err(PyValueError::new_err("pop_size must be greater than 0"));
        }
        if !is_probability(cross_rate) {
            return Err(PyValueError::new_err(format!(
                "cross_rate must be a finite probability in [0.0, 1.0], got {cross_rate}"
            )));
        }
        if !is_probability(mut_rate) {
            return Err(PyValueError::new_err(format!(
                "mut_rate must be a finite probability in [0.0, 1.0], got {mut_rate}"
            )));
        }

        let rust_graph = Graph::from_python(graph);

        if debug_level >= 1 {
            debug!(
                debug,
                "Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            rust_graph.print();
        }

        Ok(Self {
            graph: rust_graph,
            debug_level,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
        })
    }

    /// Run and return the best partition (max-Q from the Pareto front).
    /// Isolated nodes get community ``-1``.
    #[pyo3(signature = ())]
    pub fn run(&self) -> PyResult<Partition> {
        if self.is_empty() {
            return Ok(Partition::default());
        }

        let first_front: Vec<Individual> = self.rank_one_front();
        if first_front.is_empty() {
            return Ok(normalize_community_ids(&self.graph, Partition::default()));
        }

        let best_solution: &Individual = max_q_selection(&self.graph, &first_front);

        Ok(normalize_community_ids(
            &self.graph,
            best_solution.partition.clone(),
        ))
    }

    /// Return all non-dominated solutions as ``[(partition, [cut, pair]), ...]``.
    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let first_front = self.rank_one_front();
        if first_front.is_empty() {
            return Ok(Vec::new());
        }

        Ok(first_front
            .into_iter()
            .map(|ind| {
                (
                    normalize_community_ids(&self.graph, ind.partition),
                    ind.objectives,
                )
            })
            .collect())
    }

    /// Return the multi-scale profile as ``[(partition, cut, pair, gamma), ...]``,
    /// in increasing ``gamma``: the CPM resolution ladder read off one front.
    #[pyo3(signature = ())]
    pub fn ladder(&self) -> PyResult<Vec<(Partition, f64, f64, f64)>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let first_front = self.rank_one_front();
        if first_front.is_empty() {
            return Ok(Vec::new());
        }

        let (m, all_pairs) = objectives::denominators(&self.graph);

        Ok(resolution_ladder(&first_front, m, all_pairs)
            .into_iter()
            .map(|(idx, gamma)| {
                let ind = &first_front[idx];
                (
                    normalize_community_ids(&self.graph, ind.partition.clone()),
                    ind.objectives[0],
                    ind.objectives[1],
                    gamma,
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::karate::karate_club;

    type Fingerprint = (Vec<(Partition, Vec<u64>)>, Partition, Vec<(usize, u64)>);

    // one full search, every float carried as its bit pattern so equality stays exact
    fn fingerprint(graph: &Graph) -> Fingerprint {
        let front = search(graph, 12, 6, 0.7, 0.5, 0);
        assert!(!front.is_empty(), "the search returned no rank-1 member");

        let members: Vec<(Partition, Vec<u64>)> = front
            .iter()
            .map(|ind| {
                (
                    ind.partition.clone(),
                    ind.objectives.iter().map(|o| o.to_bits()).collect(),
                )
            })
            .collect();

        let selected = max_q_selection(graph, &front).partition.clone();

        let (m, all_pairs) = objectives::denominators(graph);
        let ladder: Vec<(usize, u64)> = resolution_ladder(&front, m, all_pairs)
            .into_iter()
            .map(|(idx, gamma)| (idx, gamma.to_bits()))
            .collect();

        (members, selected, ladder)
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool")
    }

    fn assert_same(a: &Fingerprint, b: &Fingerprint) {
        assert_eq!(a.0.len(), b.0.len(), "the rank-1 front changed size");
        for (i, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
            assert_eq!(x.0, y.0, "front member {i} holds a different partition");
            assert_eq!(
                x.1, y.1,
                "front member {i} holds a different objective vector"
            );
        }
        assert_eq!(a.1, b.1, "the selected partition differs");
        assert_eq!(a.2, b.2, "the resolution ladder differs");
    }

    #[test]
    fn the_whole_search_is_bit_deterministic() {
        let graph = karate_club();
        assert_same(&fingerprint(&graph), &fingerprint(&graph));
    }

    #[test]
    fn the_whole_search_ignores_the_thread_count() {
        let graph = karate_club();
        // the harness calls max_cores before every run, so the pool size must change nothing
        let one = pool(1).install(|| fingerprint(&graph));
        let many = pool(4).install(|| fingerprint(&graph));
        assert_same(&one, &many);
    }
}
