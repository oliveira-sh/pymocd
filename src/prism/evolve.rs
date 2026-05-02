//! Main PSO loop and best-solution selection.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Prism;
use super::config::{
    DEFAULT_INERTIA_HIGH, DEFAULT_INERTIA_LOW, DEFAULT_PHI_SCALE, LOCAL_OPT_EVERY, LOCAL_OPT_ITERS,
    LOCAL_OPT_TOP_N, adaptive_chi,
};
use crate::debug;
use crate::prism::archive::Archive;
use crate::prism::dense::{DenseGraph, Solution, evaluate_q_gamma, q_score};
use crate::prism::particle::{
    Particle, dense_mutate, local_optimization, maybe_update_pbest, update_particle,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

impl Prism {
    pub(super) fn envolve(
        &self,
        py: Option<Python<'_>>,
    ) -> PyResult<(DenseGraph, Vec<Solution>)> {
        let dg = DenseGraph::from_graph(&self.graph);

        let mut swarm = self.seed_swarm(&dg);

        self.evaluate_swarm(&mut swarm, &dg);
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
            let n_turb =
                (((swarm_len as f64) * self.turbulence_frac).ceil() as usize).min(swarm_len);
            if n_turb > 0 && self.mut_rate > 0.0 {
                let rate = self.mut_rate;
                swarm[..n_turb].par_iter_mut().for_each(|p| {
                    let Particle {
                        current, scratch, ..
                    } = p;
                    dense_mutate(&mut current.partition, dg_ref, rate, scratch);
                    current.objectives.clear();
                });
            }

            self.evaluate_swarm(&mut swarm, &dg);

            // Local optimization on the top-N particles by q_score every
            // LOCAL_OPT_EVERY generations. Each owns its scratch so the
            // par_iter_mut has no aliasing.
            if generation > 0 && generation % LOCAL_OPT_EVERY == 0 {
                let mut idxs: Vec<usize> = (0..swarm.len()).collect();
                idxs.sort_unstable_by(|&a, &b| {
                    q_score(&swarm[b].current)
                        .partial_cmp(&q_score(&swarm[a].current))
                        .unwrap_or(Ordering::Equal)
                });
                let top_n = LOCAL_OPT_TOP_N.min(idxs.len());
                let mut is_top: Vec<bool> = vec![false; swarm.len()];
                for &i in &idxs[..top_n] {
                    is_top[i] = true;
                }
                swarm
                    .par_iter_mut()
                    .zip(is_top.into_par_iter())
                    .for_each(|(p, top)| {
                        if !top {
                            return;
                        }
                        let Particle {
                            current, scratch, ..
                        } = p;
                        local_optimization(
                            &mut current.partition,
                            dg_ref,
                            LOCAL_OPT_ITERS,
                            scratch,
                        );
                        let objs = evaluate_q_gamma(dg_ref, &current.partition, scratch);
                        current.objectives.clear();
                        current.objectives.extend_from_slice(&objs);
                    });
            }

            for p in swarm.iter_mut() {
                maybe_update_pbest(p);
            }

            for p in swarm.iter() {
                archive.try_add(p.current.clone());
            }
            archive.prune_if_over(prune_buffer);

            if self.debug_level >= 1 && (generation % 10 == 0 || generation == self.num_gens - 1) {
                debug!(
                    debug,
                    "PRISM: Gen {} | Archive: {}/{} | chi={:.3}",
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

    pub(super) fn best_solution<'a>(&self, members: &'a [Solution]) -> &'a Solution {
        members
            .iter()
            .max_by(|a, b| q_score(a).partial_cmp(&q_score(b)).unwrap_or(Ordering::Equal))
            .expect("Empty archive")
    }
}
