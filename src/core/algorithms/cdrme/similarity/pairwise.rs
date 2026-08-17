//! The whole Eq. (4) column of one node, accumulated without materialising it.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::cdrme::topology::Topology;

/// Adds `weight * similarity(v, r)` to `acc[v]` for every `v`, Eq. (4).
///
/// Summing over `N(r)`'s rows costs `O(sum of deg(w) for w in N(r))` and touches
/// only the nodes within distance two of `r`, the only ones Eq. (4) can score.
pub fn accumulate_column(topology: &Topology, r: u32, weight: f64, acc: &mut [f64]) {
    for &w in topology.neighbors(r) {
        acc[w as usize] += weight;
        for &x in topology.neighbors(w) {
            acc[x as usize] += weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_matches_the_pairwise_equation() {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)];
        let topology = Topology::from_edges(&nodes, &edges);
        for &r in &topology.active {
            let mut acc = vec![0.0; topology.n];
            accumulate_column(&topology, r, 2.0, &mut acc);
            for &v in &topology.active {
                assert!(
                    (acc[v as usize] - 2.0 * f64::from(topology.similarity(v, r))).abs() < 1e-12,
                    "sim({v},{r})"
                );
            }
        }
    }
}
