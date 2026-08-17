//! Remapping an internal partition to the crate's output labels, with isolated
//! nodes reported as -1.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::super::Labels;

pub fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
    let mut remap = vec![-1i32; g.n];
    let mut next = 0i32;
    let mut out = Vec::with_capacity(g.n);
    for (i, &d) in g.deg.iter().enumerate() {
        let comm = if d == 0 {
            -1
        } else {
            let slot = &mut remap[labels[i] as usize];
            if *slot < 0 {
                *slot = next;
                next += 1;
            }
            *slot
        };
        out.push((g.labels[i], comm));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::CsrGraph;

    #[test]
    fn isolated_nodes_report_minus_one_and_ids_are_dense() {
        let g = CsrGraph::from_edges(&[10, 11, 12, 13], &[(10, 11), (12, 13)]);
        let out = to_output(&g, &vec![3, 3, 1, 1]);
        assert_eq!(out, vec![(10, 0), (11, 0), (12, 1), (13, 1)]);

        let g = CsrGraph::from_edges(&[0, 1, 2], &[(0, 1)]);
        let out = to_output(&g, &vec![0, 0, 2]);
        assert_eq!(out[2], (2, -1), "the isolated node kept a community");
    }
}
