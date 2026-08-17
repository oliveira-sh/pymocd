//! Macro variation: uniform crossover plus bit-flip mutation over the centre
//! indicator genome.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rayon::prelude::*;

use crate::core::algorithms::smocc::Genome;
use crate::core::algorithms::smocc::utils::sampling::{bernoulli, slot_rng, tournament};

/// `mac_mode` selects the mutation.
///
/// `0` is the flat per-bit flip: every position of the crossover child flips
/// with probability `p_m`. At `p_m = 0.5` the child is a uniform random genome,
/// independent of both parents, with `n/2` centres.
///
/// `1` is the centre-preserving flip: a centre is dropped with probability
/// `p_m` and the same expected number of new centres is drawn from the
/// non-centres, so `p_m` is the fraction of centres resampled per generation
/// and the expected centre count is unchanged.
pub fn macro_offspring_mode(
    parents: &[Genome],
    ranks: &[usize],
    crowd: &[f64],
    p_m: f64,
    salt: u64,
    mac_mode: u8,
) -> Vec<Genome> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = parents[0].len();
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let b = tournament(ranks, crowd, &mut r);
            let (pa, pb) = (&parents[a], &parents[b]);

            let d_half = bernoulli(0.5);
            let mut child: Genome = Vec::with_capacity(n);
            for i in 0..n {
                child.push(if r.sample(d_half) { pa[i] } else { pb[i] });
            }

            if mac_mode == 1 {
                let c = child.iter().filter(|&&b| b != 0).count();
                let p_on = p_m.clamp(0.0, 1.0);
                let p_off = if n > c {
                    (p_on * c as f64 / (n - c) as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let d_on = bernoulli(p_on);
                let d_off = bernoulli(p_off);
                for bit in &mut child {
                    let d = if *bit != 0 { d_on } else { d_off };
                    if r.sample(d) {
                        *bit ^= 1;
                    }
                }
            } else {
                let d_m = bernoulli(p_m);
                for bit in &mut child {
                    if r.sample(d_m) {
                        *bit ^= 1;
                    }
                }
            }

            if child.iter().all(|&b| b == 0) && n > 0 {
                let k = r.random_range(0..n);
                child[k] = 1;
            }
            child
        })
        .collect()
}
