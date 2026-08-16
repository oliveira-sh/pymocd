//! Initial population construction for the micro (neighbour-label) and macro
//! (degree-biased center) swarms.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::core::graph::CsrGraph;

use super::Labels;
use super::config::defaults::DEFAULT_MACRO_CAP;
use super::config::objectives::Cfg;
use super::operators;
use super::particles::{Mac, Mic};
use super::sim::decode;

const MACRO_CAP_MIN: f64 = 1e-6;
const MACRO_CAP_MAX: f64 = 1e6;

pub(super) fn macro_cmax(n: usize, macro_cap: f64) -> usize {
    let mult = if macro_cap.is_nan() {
        DEFAULT_MACRO_CAP
    } else {
        macro_cap.clamp(MACRO_CAP_MIN, MACRO_CAP_MAX)
    };
    ((mult * (n as f64).sqrt()).ceil() as usize).clamp(1, n.max(1))
}

pub(super) fn init_micro(g: &CsrGraph, pop: usize, cfg: &Cfg) -> Vec<Mic> {
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = operators::slot_rng(u64::MAX, k);
            let labels: Labels = (0..g.n)
                .map(|i| {
                    let nbrs = g.neighbors(i);
                    if nbrs.is_empty() {
                        i as i32
                    } else {
                        nbrs[r.random_range(0..nbrs.len())] as i32
                    }
                })
                .collect();
            let obj = cfg.eval_micro(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

pub(super) fn init_macro(
    g: &CsrGraph,
    wadj: &[f64],
    pop: usize,
    cfg: &Cfg,
    macro_cap: f64,
) -> Vec<Mac> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].cmp(&g.deg[a]));
    let cmax = macro_cmax(n, macro_cap);
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = operators::slot_rng(u64::MAX - 1, k);
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
            let labels = decode(g, wadj, &genome);
            let obj = cfg.eval_macro(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
        })
        .collect()
}
