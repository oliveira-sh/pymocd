//! Macro-side offspring: uniform crossover plus per-bit mutation of the genome.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;

use crate::core::algorithms::mmcomo::Genome;

use super::mating::tournament;

/// Macro offspring (Alg. 1 line 5): uniform crossover + per-bit mutation.
pub fn macro_offspring(
    parents: &[Genome],
    ranks: &[usize],
    crowd: &[f64],
    p_m: f64,
) -> Vec<Genome> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = parents[0].len();
    let mut r = rand::rng();
    let mut children: Vec<Genome> = Vec::with_capacity(pop);

    for _ in 0..pop {
        let a = tournament(ranks, crowd, &mut r);
        let b = tournament(ranks, crowd, &mut r);
        let (pa, pb) = (&parents[a], &parents[b]);

        let mut child: Genome = Vec::with_capacity(n);
        for i in 0..n {
            let mut bit = if r.random_bool(0.5) { pa[i] } else { pb[i] };
            if r.random_bool(p_m) {
                bit ^= 1;
            }
            child.push(bit);
        }

        // an all-zero genome would decode through decode()'s max-degree fallback,
        // collapsing the child to one community; a drawn centre keeps the diversity
        if child.iter().all(|&b| b == 0) && n > 0 {
            let k = r.random_range(0..n);
            child[k] = 1;
        }
        children.push(child);
    }
    children
}
