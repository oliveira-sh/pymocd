//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mr_mocd::Labels;
use crate::core::algorithms::mr_mocd::objectives::{Obj, community_count};
use crate::core::graph::CsrGraph;

use super::codelength::shortest_code;
use super::modularity::max_modularity;
use super::plateau::widest_plateau;

const DEGENERATE_FRACTION: f64 = 0.5;

pub fn counts(g: &CsrGraph, front: &[Labels]) -> Vec<usize> {
    front
        .par_iter()
        .map_init(|| vec![false; g.n], |seen, p| community_count(p, seen))
        .collect()
}

fn lambda_span(g: &CsrGraph) -> (f64, f64) {
    let n = g.n as f64;
    let density = 2.0 * g.m as f64 / (n * (n - 1.0));
    if density <= 0.0 || density.is_nan() {
        return (0.0, f64::INFINITY);
    }
    (1.0 / (n * n * density), 1.0 / density)
}

pub fn select_index(g: &CsrGraph, front: &[Labels], objs: &[Obj]) -> usize {
    let k = counts(g, front);
    let cap = (DEGENERATE_FRACTION * g.n as f64).max(2.0);
    let mut keep: Vec<usize> = (0..front.len())
        .filter(|&i| k[i] > 1 && (k[i] as f64) < cap)
        .collect();
    if keep.is_empty() {
        keep = (0..front.len()).collect();
    }

    shortest_code(g, front, &keep)
        .or_else(|| widest_plateau(objs, &k, &keep, lambda_span(g)))
        .unwrap_or_else(|| max_modularity(g, front, objs, &keep))
}

pub fn select_best(g: &CsrGraph, front: Vec<Labels>, objs: &[Obj]) -> Labels {
    let pick = select_index(g, &front, objs);
    front.into_iter().nth(pick).unwrap()
}
