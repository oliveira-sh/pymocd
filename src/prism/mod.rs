//! prism/mod.rs
//! PRISM — Pareto Resolution-Invariant Swarm for Modularity.
//! Multi-objective PSO for community detection:
//!   - Q_γ multi-resolution Pareto front (γ = 0.5, 1.0, 2.0) over
//!     TOM/Jaccard-reweighted edges
//!   - EM-momentum velocity with adaptive Clerc-Kennedy constriction
//!   - LPA-seeded swarm + memetic Louvain refinement on top Pareto particles
//!   - Non-dominated archive with crowding-distance leader selection
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod archive;
pub(crate) mod dense;
mod particle;

use crate::debug;
use crate::graph::{CommunityId, Graph, Partition};
use crate::utils::normalize_community_ids;

use archive::Archive;
use dense::{
    DenseGraph, DensePartition, Solution, crowding_distance, evaluate_q_gamma, lpa_partition,
    q_score,
};
use particle::{
    Particle, dense_mutate, louvain_refine, maybe_update_pbest, update_particle,
};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rand::RngExt;
use rayon::prelude::*;
use std::cmp::Ordering;

const DEFAULT_INERTIA_LOW: f64 = 0.1;
const DEFAULT_INERTIA_HIGH: f64 = 0.4;
const DEFAULT_V_MAX: f64 = 0.3;
const DEFAULT_PHI_SCALE: f64 = 0.1;
const DEFAULT_INITIAL_V: f64 = 0.1;
const DEFAULT_BETA: f64 = 0.7;
const DEFAULT_LPA_FRAC: f64 = 0.7;
const DEFAULT_LPA_ITERS: usize = 3;
const DEFAULT_VNR_FRAC: f64 = 0.15;

/// Clerc-Kennedy constriction: χ = 2 / |2 - φ - √(φ² - 4φ)|, with φ > 4.
fn constriction(phi: f64) -> f64 {
    if phi <= 4.0 {
        return 1.0;
    }
    let disc = (phi * phi - 4.0 * phi).sqrt();
    2.0 / (2.0 - phi - disc).abs()
}

#[pyclass]
pub struct Prism {
    graph: Graph,
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
    #[allow(dead_code)]
    vnr_frac: f64,
    py_graph: Option<Py<PyAny>>,
    py_objectives: Vec<Py<PyAny>>,
    on_generation: Option<Py<PyAny>>,
}

fn random_dense_partition(n: usize) -> DensePartition {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| rng.random_range(0..n as CommunityId))
        .collect()
}

/// Perturb an LPA partition by randomly relabelling `frac` fraction of nodes.
/// Keeps topology signal while injecting diversity across the swarm.
fn perturb_partition(p: &[CommunityId], frac: f64) -> DensePartition {
    let mut rng = rand::rng();
    let n = p.len();
    let mut out = p.to_vec();
    for j in 0..n {
        if rng.random::<f64>() < frac {
            out[j] = rng.random_range(0..n as CommunityId);
        }
    }
    out
}

impl Prism {
    fn evaluate_swarm(
        &self,
        py: Option<Python<'_>>,
        swarm: &mut [Particle],
        dg: &DenseGraph,
    ) -> PyResult<()> {
        if self.py_objectives.is_empty() {
            swarm.par_iter_mut().for_each(|p| {
                let objs = evaluate_q_gamma(dg, &p.current.partition);
                p.current.objectives.clear();
                p.current.objectives.extend_from_slice(&objs);
            });
            return Ok(());
        }
        let py = py.expect("Python token required when py_objectives are set");
        let py_graph = self
            .py_graph
            .as_ref()
            .expect("py_graph must be set when py_objectives are used");
        let py_objs = &self.py_objectives;
        let partition_dict = PyDict::new(py);
        let graph_ref = py_graph.bind(py);
        for p in swarm.iter_mut() {
            partition_dict.clear();
            for (i, &comm) in p.current.partition.iter().enumerate() {
                partition_dict.set_item(dg.nodes[i], comm)?;
            }
            let mut objectives = Vec::with_capacity(py_objs.len());
            for obj in py_objs.iter() {
                let value = obj
                    .bind(py)
                    .call1((graph_ref, &partition_dict))?
                    .extract::<f64>()?;
                objectives.push(value);
            }
            p.current.objectives = objectives;
        }
        Ok(())
    }

    fn seed_swarm(&self, dg: &DenseGraph) -> Vec<Particle> {
        let n_nodes = dg.n;
        let initial_v = DEFAULT_INITIAL_V.min(self.v_max);
        let n_lpa = ((self.swarm_size as f64) * self.lpa_frac).round() as usize;
        let n_lpa = n_lpa.min(self.swarm_size);
        // Build a canonical LPA seed once, then perturb for diversity.
        // Guard against LPA collapse at high μ: if unique-label count is tiny,
        // fall back to random partitions for this batch.
        let mut lpa_seed = if n_lpa > 0 {
            lpa_partition(dg, self.lpa_iters)
        } else {
            Vec::new()
        };
        if !lpa_seed.is_empty() {
            use rustc_hash::FxHashSet;
            let uniq: FxHashSet<CommunityId> = lpa_seed.iter().copied().collect();
            let min_k = (dg.n as f64).sqrt().floor() as usize / 2;
            if uniq.len() < min_k.max(4) {
                lpa_seed.clear();
            }
        }

        let mut swarm: Vec<Particle> = Vec::with_capacity(self.swarm_size);

        let effective_lpa = if lpa_seed.is_empty() { 0 } else { n_lpa };
        let lpa_batch: Vec<Particle> = (0..effective_lpa)
            .into_par_iter()
            .map(|i| {
                let part = if i == 0 {
                    lpa_seed.clone()
                } else {
                    // increasing perturbation so swarm spreads, head stays clean
                    let frac = 0.02 + 0.08 * (i as f64 / (effective_lpa.max(1) as f64));
                    perturb_partition(&lpa_seed, frac)
                };
                Particle::new(part, initial_v)
            })
            .collect();
        swarm.extend(lpa_batch);

        let n_remaining = self.swarm_size - effective_lpa;
        let rand_batch: Vec<Particle> = (0..n_remaining)
            .into_par_iter()
            .map(|_| Particle::new(random_dense_partition(n_nodes), initial_v))
            .collect();
        swarm.extend(rand_batch);
        swarm
    }

    fn envolve(&self, py: Option<Python<'_>>) -> PyResult<(DenseGraph, Vec<Solution>)> {
        let dg = DenseGraph::from_graph(&self.graph);

        let mut swarm = self.seed_swarm(&dg);

        self.evaluate_swarm(py, &mut swarm, &dg)?;
        for p in swarm.iter_mut() {
            p.pbest = p.current.clone();
        }

        let mut archive = Archive::new(self.archive_cap);
        for p in swarm.iter() {
            archive.try_add(p.current.clone());
        }
        archive.prune();

        let prune_buffer = self.archive_cap / 4;
        let v_max = self.v_max;
        let beta = self.beta;
        let dg_ref = &dg;

        for generation in 0..self.num_gens {
            if archive.len() == 0 {
                if let Some(p) = swarm.first() {
                    archive.try_add(p.current.clone());
                    archive.refresh_crowding();
                }
            }

            // Constriction Fairness: adapt χ from archive diversity.
            // High avg CD → exploratory archive, boost χ to exploit; low CD →
            // collapsed archive, shrink χ to diversify.
            let chi = adaptive_chi(&archive, generation, self.num_gens);

            swarm.par_iter_mut().for_each(|p| {
                let leader = archive.select_leader();
                update_particle(
                    p,
                    leader,
                    dg_ref,
                    DEFAULT_INERTIA_LOW,
                    DEFAULT_INERTIA_HIGH,
                    v_max,
                    DEFAULT_PHI_SCALE,
                    beta,
                    chi,
                );
            });

            let swarm_len = swarm.len();
            let n_turb = (((swarm_len as f64) * self.turbulence_frac).ceil() as usize)
                .min(swarm_len);
            if n_turb > 0 && self.mut_rate > 0.0 {
                swarm[..n_turb].par_iter_mut().for_each(|p| {
                    dense_mutate(&mut p.current.partition, dg_ref, self.mut_rate);
                });
            }

            self.evaluate_swarm(py, &mut swarm, &dg)?;

            // Memetic Leiden-style local search on top Pareto particles.
            // Every MEMETIC_EVERY generations, take top-N by q_score, apply
            // louvain_refine (1 iter) to escape modularity local optima.
            const MEMETIC_EVERY: usize = 5;
            const MEMETIC_TOP_N: usize = 10;
            const MEMETIC_ITERS: usize = 2;
            if generation > 0 && generation % MEMETIC_EVERY == 0 {
                let mut idxs: Vec<usize> = (0..swarm.len()).collect();
                idxs.sort_by(|&a, &b| {
                    q_score(&swarm[b].current)
                        .partial_cmp(&q_score(&swarm[a].current))
                        .unwrap_or(Ordering::Equal)
                });
                let top: Vec<usize> = idxs.into_iter().take(MEMETIC_TOP_N).collect();
                for i in top {
                    louvain_refine(&mut swarm[i].current.partition, dg_ref, MEMETIC_ITERS);
                    let objs = evaluate_q_gamma(dg_ref, &swarm[i].current.partition);
                    swarm[i].current.objectives.clear();
                    swarm[i].current.objectives.extend_from_slice(&objs);
                }
            }

            for p in swarm.iter_mut() {
                maybe_update_pbest(p);
            }

            for p in swarm.iter() {
                archive.try_add(p.current.clone());
            }
            archive.prune_if_over(prune_buffer);

            if self.debug_level >= 1
                && (generation % 10 == 0 || generation == self.num_gens - 1)
            {
                debug!(
                    debug,
                    "PRISM: Gen {} | Archive: {}/{} | χ={:.3}",
                    generation,
                    archive.len(),
                    self.archive_cap,
                    chi
                );
            }

            if let Some(cb) = &self.on_generation {
                if let Some(py) = py {
                    cb.bind(py)
                        .call1((generation, self.num_gens, archive.len()))?;
                }
            }
        }

        Ok((dg, archive.members))
    }

    fn best_solution<'a>(&self, members: &'a [Solution]) -> &'a Solution {
        members
            .iter()
            .max_by(|a, b| q_score(a).partial_cmp(&q_score(b)).unwrap_or(Ordering::Equal))
            .expect("Empty archive")
    }
}

/// Adaptive constriction χ.
/// Baseline φ = c1+c2 in [3, 4.1]; pick φ dynamically:
///   early generations → exploration (lower χ)
///   late generations  → exploitation (higher χ bounded)
/// Also nudge by archive avg crowding distance.
fn adaptive_chi(archive: &Archive, generation: usize, total_gens: usize) -> f64 {
    let t = if total_gens <= 1 {
        1.0
    } else {
        generation as f64 / (total_gens as f64 - 1.0)
    };
    // φ ramps from 4.05 → 4.2 over the run.
    let phi = 4.05 + 0.15 * t;
    let base = constriction(phi);

    // Diversity nudge: if archive small or collapsed, shrink χ by 10%.
    let diverse = archive.len() >= 3;
    if diverse { base } else { base * 0.9 }
}

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
            vnr_frac: DEFAULT_VNR_FRAC,
            py_graph: None,
            py_objectives: vec![],
            on_generation: None,
        }
    }

    pub fn _run(&self) -> Partition {
        let (dg, front) = self.envolve(None).expect("envolve failed");
        let best = self.best_solution(&front);
        normalize_community_ids(&self.graph, dg.sparse_partition(&best.partition))
    }
}

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
        vnr_frac = DEFAULT_VNR_FRAC,
        objectives = None
    ))]
    pub fn new(
        _py: Python<'_>,
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
        vnr_frac: f64,
        objectives: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python(graph);

        if debug_level >= 1 {
            debug!(
                debug,
                "PRISM Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            rust_graph.print();
        }

        let py_graph = Some(graph.clone().unbind());
        let py_objectives: Vec<Py<PyAny>> = objectives
            .map(|obj_list| obj_list.iter().map(|item| item.unbind()).collect())
            .unwrap_or_default();

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
            vnr_frac,
            py_graph,
            py_objectives,
            on_generation: None,
        })
    }

    #[pyo3(signature = (objectives))]
    pub fn set_objectives(&mut self, objectives: &Bound<'_, PyList>) -> PyResult<()> {
        self.py_objectives = objectives.iter().map(|item| item.unbind()).collect();
        Ok(())
    }

    #[pyo3(signature = (callback))]
    pub fn set_on_generation(&mut self, callback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.on_generation = callback.map(|cb| cb.clone().unbind());
        Ok(())
    }

    #[getter]
    pub fn num_gens(&self) -> usize {
        self.num_gens
    }

    #[pyo3(signature = ())]
    pub fn generate_pareto_front(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let (dg, front) = self.envolve(Some(py))?;
        Ok(front
            .into_iter()
            .map(|sol| {
                let sparse = dg.sparse_partition(&sol.partition);
                (normalize_community_ids(&self.graph, sparse), sol.objectives)
            })
            .collect())
    }

    #[pyo3(signature = (polish_iters = 20))]
    pub fn run(&self, py: Python<'_>, polish_iters: usize) -> PyResult<Partition> {
        let (dg, front) = self.envolve(Some(py))?;
        let best = self.best_solution(&front);
        let mut refined = best.partition.clone();
        if polish_iters > 0 {
            louvain_refine(&mut refined, &dg, polish_iters);
        }
        Ok(normalize_community_ids(
            &self.graph,
            dg.sparse_partition(&refined),
        ))
    }

    #[pyo3(signature = ())]
    pub fn best_q(&self, py: Python<'_>) -> PyResult<f64> {
        let (_, front) = self.envolve(Some(py))?;
        Ok(q_score(self.best_solution(&front)))
    }
}

#[allow(dead_code)]
fn _refresh_cd(archive: &mut Archive) {
    crowding_distance(&mut archive.members);
}
