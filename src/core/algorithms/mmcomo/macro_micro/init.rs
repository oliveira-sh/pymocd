//! Initial micro and macro populations (Alg. 1 lines 1-2).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::collections::HashSet;

use rand::seq::SliceRandom;
use rand::{RngExt, rng};

use crate::core::algorithms::mmcomo::objectives::kkm_rc;
use crate::core::algorithms::mmcomo::similarity::decode;
use crate::core::algorithms::mmcomo::{Graph, Labels, Sm};

use super::swarms::{Mac, Mic};

/// Micro init (Alg. 1 line 1): each node's label = a random neighbour's id.
pub fn init_micro(g: &Graph, pop: usize) -> Vec<Mic> {
    (0..pop)
        .map(|_| {
            let mut r = rng();
            let labels: Labels = (0..g.n)
                .map(|i| {
                    if g.adj[i].is_empty() {
                        i as i32
                    } else {
                        g.adj[i][r.random_range(0..g.adj[i].len())] as i32
                    }
                })
                .collect();
            let obj = kkm_rc(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

/// Macro init (Alg. 1 line 2; ref [46]): half high-degree seeded, half random.
/// Centre count and pool size are NOT pinned by the paper (deferred to ref [46]):
/// centre count in `[1, ⌈√n⌉]`, high-degree half sampled from the top-`3c`.
pub fn init_macro(g: &Graph, sm: &Sm, pop: usize) -> Vec<Mac> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].partial_cmp(&g.deg[a]).unwrap());
    let cmax = ((n as f64).sqrt().ceil() as usize).clamp(1, n);
    let mut r = rng();
    (0..pop)
        .map(|k| {
            let c = r.random_range(1..=cmax);
            let mut genome = vec![0u8; n];
            if k < pop / 2 {
                let cand = (3 * c).min(n);
                let mut poolv: Vec<usize> = by_deg[..cand].to_vec();
                poolv.shuffle(&mut r);
                for &i in poolv.iter().take(c) {
                    genome[i] = 1;
                }
            } else {
                let mut chosen: HashSet<usize> = HashSet::new();
                while chosen.len() < c {
                    chosen.insert(r.random_range(0..n));
                }
                for i in chosen {
                    genome[i] = 1;
                }
            }
            if genome.iter().all(|&b| b == 0) {
                genome[by_deg[0]] = 1;
            }
            let labels = decode(g, sm, &genome);
            let obj = kkm_rc(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
        })
        .collect()
}
