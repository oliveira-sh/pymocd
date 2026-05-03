//! Swarm-wide objective evaluation.
//!
//! Parallel `evaluate_q_gamma` with a dirty-flag skip: particles whose
//! `current.objectives` is non-empty are still in sync with their
//! partition and bypass re-evaluation.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Prism;
use crate::core::prism::dense::{DenseGraph, evaluate_q_gamma};
use crate::core::prism::particle::Particle;
use rayon::prelude::*;

impl Prism {
    pub(super) fn evaluate_swarm(&self, swarm: &mut [Particle], dg: &DenseGraph) {
        swarm.par_iter_mut().for_each(|p| {
            let Particle {
                current, scratch, ..
            } = p;
            if !current.objectives.is_empty() {
                return;
            }
            let objs = evaluate_q_gamma(dg, &current.partition, scratch);
            current.objectives.extend_from_slice(&objs);
        });
    }
}
