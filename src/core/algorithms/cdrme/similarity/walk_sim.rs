//! Eq. (8), rescaled onto the [0,1] range every downstream unit assumes.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::pairwise::accumulate_column;
use crate::core::algorithms::cdrme::topology::Topology;

/// Fills `out` with `AvgSimilarity(v, primWalk)` (Eq. 8) divided by its own
/// maximum, and pins the centre's own gene to 1.0.
///
/// Eq. (8) is an unbounded count; the rescale is a deliberate divergence that
/// Figs. 4-5 and the Sec. 4.4.2 threshold both require (README, "Divergences").
pub fn avg_similarity(topology: &Topology, center: u32, prim: &[(u32, u32)], out: &mut [f64]) {
    out.fill(0.0);
    let mut frequency = 0.0;
    for &(node, freq) in prim {
        frequency += f64::from(freq);
        accumulate_column(topology, node, f64::from(freq), out);
    }
    if frequency > 0.0 {
        for value in out.iter_mut() {
            *value /= frequency;
        }
    }

    let mut peak = 0.0f64;
    for &v in &topology.active {
        peak = peak.max(out[v as usize]);
    }
    if peak > 0.0 {
        for &v in &topology.active {
            out[v as usize] /= peak;
        }
    }
    out[center as usize] = 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_centre_leads_its_own_walk_and_nothing_exceeds_one() {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)];
        let topology = Topology::from_edges(&nodes, &edges);
        let mut out = vec![0.0; topology.n];
        avg_similarity(&topology, 0, &[(0, 3), (1, 2), (2, 2)], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!(topology.active.iter().all(|&v| out[v as usize] <= 1.0));
        assert!(out[1] > out[4], "{out:?}");
        assert_eq!(out[6], 0.0, "an isolated node scores 0 against every walk");
    }

    #[test]
    fn an_empty_walk_leaves_only_the_centre() {
        let topology = Topology::from_edges(&[0, 1], &[(0, 1)]);
        let mut out = vec![0.0; topology.n];
        avg_similarity(&topology, 1, &[], &mut out);
        assert_eq!(out, vec![0.0, 1.0]);
    }
}
