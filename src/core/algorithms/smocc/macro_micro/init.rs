//! Initial population construction for the micro (neighbour-label) and macro
//! (degree-biased center) swarms.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::config::Cfg;
use crate::core::algorithms::smocc::config::defaults::DEFAULT_MACRO_CAP;
use crate::core::algorithms::smocc::similarity::decode;
use crate::core::algorithms::smocc::utils::sampling::slot_rng;
use crate::core::graph::CsrGraph;

use super::swarms::{Mac, Mic};

const MACRO_CAP_MIN: f64 = 1e-6;

const MACRO_CAP_MAX: f64 = 1e6;

pub fn macro_cmax(n: usize, macro_cap: f64) -> usize {
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
            let mut r = slot_rng(u64::MAX, k);
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
            let mut r = slot_rng(u64::MAX - 1, k);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::config::defaults::DEFAULT_MACRO_CAP;

    #[test]
    fn macro_cmax_default_reproduces_hardcoded_sqrt_ceiling() {
        for n in (1..600usize).chain([1000, 1999, 2000, 5000, 9999, 10_000, 65_536]) {
            let old = ((n as f64).sqrt().ceil() as usize).clamp(1, n);
            assert_eq!(
                macro_cmax(n, DEFAULT_MACRO_CAP),
                old,
                "n={n}: macro_cap=1.0 diverged from the pre-change ceiling"
            );
        }
        assert_eq!(DEFAULT_MACRO_CAP, 1.0);
    }

    #[test]
    fn macro_cmax_table() {
        assert_eq!(macro_cmax(250, 1.0), 16);
        assert_eq!(macro_cmax(2000, 1.0), 45);
        assert_eq!(macro_cmax(10_000, 1.0), 100);
        assert_eq!(macro_cmax(10_000, 4.0), 400);
        assert_eq!(macro_cmax(10_000, 1e9), 10_000);
        assert_eq!(macro_cmax(10_000, f64::INFINITY), 10_000);
        assert_eq!(macro_cmax(100, 50.0), 100);
        for bad in [0.0, -1.0, -1e30, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                macro_cmax(10_000, bad) >= 1,
                "macro_cap={bad} produced a zero ceiling"
            );
            assert!(macro_cmax(1, bad) >= 1);
        }
        assert_eq!(macro_cmax(0, 1.0), 1);
    }
}
