//! Initial swarm construction.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Prism;
use super::config::{DEFAULT_INITIAL_V, perturb_partition, random_dense_partition};
use crate::graph::CommunityId;
use crate::prism::dense::{DenseGraph, lpa_partition};
use crate::prism::particle::Particle;
use rayon::prelude::*;

impl Prism {
    pub(super) fn seed_swarm(&self, dg: &DenseGraph) -> Vec<Particle> {
        let n_nodes = dg.n;
        let initial_v = DEFAULT_INITIAL_V.min(self.v_max);
        let n_lpa = ((self.swarm_size as f64) * self.lpa_frac).round() as usize;
        let n_lpa = n_lpa.min(self.swarm_size);
        let mut lpa_seed = if n_lpa > 0 {
            lpa_partition(dg, self.lpa_iters)
        } else {
            Vec::new()
        };
        // Drop LPA seed if it collapsed to a near-single community.
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
}
