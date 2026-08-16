//! The HP-MOCD modularity decomposition into an intra-edge deficit and an
//! inter-community degree mass.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc::Labels;
use crate::core::graph::CsrGraph;

use super::UNSEEN;

pub fn intra_inter(g: &CsrGraph, labels: &Labels) -> (f64, f64) {
    let m = g.m as f64;
    if m == 0.0 {
        return (0.0, 0.0);
    }
    let two_m = 2.0 * m;

    let mut slot: Vec<u32> = vec![UNSEEN; g.n];
    let mut d_c: Vec<f64> = Vec::new();
    for (&c, &k) in labels.iter().zip(g.deg.iter()) {
        debug_assert!((c as usize) < g.n, "label {c} outside [0,{})", g.n);
        let s = slot[c as usize];
        let b = if s == UNSEEN {
            let b = d_c.len() as u32;
            slot[c as usize] = b;
            d_c.push(0.0);
            b
        } else {
            s
        } as usize;
        d_c[b] += f64::from(k);
    }

    let mut l_intra = 0.0f64;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            l_intra += 1.0;
        }
    }

    let inter: f64 = d_c.iter().map(|&d| (d / two_m).powi(2)).sum();
    (1.0 - l_intra / m, inter)
}
