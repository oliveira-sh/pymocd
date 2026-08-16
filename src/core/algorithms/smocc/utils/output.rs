//! Remapping an internal partition to the crate's output labels, with isolated
//! nodes reported as -1.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::graph::CsrGraph;

use super::super::Labels;

pub fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
    let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(g.n);
    for (i, &d) in g.deg.iter().enumerate() {
        let comm = if d == 0 {
            -1
        } else {
            *remap.entry(labels[i]).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        };
        out.push((g.labels[i], comm));
    }
    out
}
