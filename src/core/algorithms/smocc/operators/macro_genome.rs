//! Macro variation: uniform crossover plus bit-flip mutation over the centre
//! indicator genome.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rayon::prelude::*;

use crate::core::algorithms::smocc::Genome;
use crate::core::algorithms::smocc::utils::sampling::{bernoulli, slot_rng, tournament};

pub fn macro_offspring(
    parents: &[Genome],
    ranks: &[usize],
    crowd: &[f64],
    p_m: f64,
    salt: u64,
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
            let d_m = if n > 0 { bernoulli(p_m) } else { d_half };

            let mut child: Genome = Vec::with_capacity(n);
            for i in 0..n {
                let mut bit = if r.sample(d_half) { pa[i] } else { pb[i] };
                if r.sample(d_m) {
                    bit ^= 1;
                }
                child.push(bit);
            }

            if child.iter().all(|&b| b == 0) && n > 0 {
                let k = r.random_range(0..n);
                child[k] = 1;
            }
            child
        })
        .collect()
}
