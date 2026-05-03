//! Python-facing methods for `Prism` plus native Rust ctors.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Prism;
use super::config::{DEFAULT_BETA, DEFAULT_LPA_FRAC, DEFAULT_LPA_ITERS, DEFAULT_V_MAX};
use crate::core::graph::{Graph, Partition};
use crate::core::prism::dense::Scratch;
use crate::core::prism::particle::local_optimization;
use crate::core::utils::normalize_community_ids;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::gen_stub_pymethods;

impl Prism {
    pub fn _new(graph: Graph) -> Self {
        Prism {
            graph,
            debug_level: 0,
            swarm_size: 100,
            num_gens: 100,
            archive_cap: 100,
            mut_rate: 0.2,
            turbulence_frac: 0.15,
            v_max: DEFAULT_V_MAX,
            beta: DEFAULT_BETA,
            lpa_frac: DEFAULT_LPA_FRAC,
            lpa_iters: DEFAULT_LPA_ITERS,
            on_generation: None,
        }
    }

    pub fn _run(&self) -> Partition {
        let (dg, front) = self.envolve(None).expect("envolve failed");
        let best = self.best_solution(&front);
        normalize_community_ids(&self.graph, dg.sparse_partition(&best.partition))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Prism {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = 0,
        swarm_size = 100,
        num_gens = 100,
        archive_cap = 100,
        mut_rate = 0.2,
        turbulence_frac = 0.15,
        v_max = DEFAULT_V_MAX,
        beta = DEFAULT_BETA,
        lpa_frac = DEFAULT_LPA_FRAC,
        lpa_iters = DEFAULT_LPA_ITERS,
    ))]
    pub fn new(
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        swarm_size: usize,
        num_gens: usize,
        archive_cap: usize,
        mut_rate: f64,
        turbulence_frac: f64,
        v_max: f64,
        beta: f64,
        lpa_frac: f64,
        lpa_iters: usize,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python_any(py, graph)?;

        if debug_level >= 1 {
            debug!(
                debug,
                "PRISM Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            rust_graph.print();
        }

        Ok(Prism {
            graph: rust_graph,
            debug_level,
            swarm_size,
            num_gens,
            archive_cap,
            mut_rate,
            turbulence_frac,
            v_max,
            beta,
            lpa_frac,
            lpa_iters,
            on_generation: None,
        })
    }

    /// Register per-generation callback ``(gen, num_gens, archive_size) -> None``.
    /// Pass ``None`` to clear.
    #[pyo3(signature = (callback))]
    pub fn set_on_generation(&mut self, callback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.on_generation = callback.map(|cb| cb.clone().unbind());
        Ok(())
    }

    /// Configured number of generations.
    #[getter]
    pub fn num_gens(&self) -> usize {
        self.num_gens
    }

    /// Return all non-dominated solutions as ``[(partition, [intra, inter]), ...]``.
    #[pyo3(signature = ())]
    pub fn generate_pareto_front(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(Py<PyDict>, Vec<f64>)>> {
        let (dg, front) = self.envolve(Some(py))?;
        let mut out = Vec::with_capacity(front.len());
        for sol in front.into_iter() {
            let sparse = dg.sparse_partition(&sol.partition);
            let normalized = normalize_community_ids(&self.graph, sparse);
            out.push((self.graph.py_partition(py, &normalized)?, sol.objectives));
        }
        Ok(out)
    }

    /// Run and return the best partition. ``polish_iters`` rounds of local
    /// search are applied (0 to skip). Isolated nodes get community ``-1``.
    #[pyo3(signature = (polish_iters = 20))]
    pub fn run(&self, py: Python<'_>, polish_iters: usize) -> PyResult<Py<PyDict>> {
        let (dg, front) = self.envolve(Some(py))?;
        let best = self.best_solution(&front);
        let mut refined = best.partition.clone();
        if polish_iters > 0 {
            let mut scratch = Scratch::new(dg.n);
            local_optimization(&mut refined, &dg, polish_iters, &mut scratch);
        }
        let normalized = normalize_community_ids(&self.graph, dg.sparse_partition(&refined));
        self.graph.py_partition(py, &normalized)
    }

    /// Run and return modularity Q of the best solution (``1 - intra - inter``).
    /// Use ``run()`` if you also need the partition.
    #[pyo3(signature = ())]
    pub fn best_q(&self, py: Python<'_>) -> PyResult<f64> {
        let (_, front) = self.envolve(Some(py))?;
        Ok(super::dense::q_score(self.best_solution(&front)))
    }
}
