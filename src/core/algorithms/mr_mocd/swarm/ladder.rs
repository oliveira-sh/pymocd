//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

pub fn ladder(g: &CsrGraph, pop: usize) -> Vec<f64> {
    debug_assert!(pop >= 2, "the swarm is at least two particles");
    let n = g.n;
    if n < 2 || g.m == 0 {
        return vec![1.0; pop];
    }
    let n = n as f64;
    (0..pop)
        .map(|k| {
            let t = k as f64 / (pop - 1) as f64;
            n.powf(2.0 * t - 2.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mr_mocd::utils::fixtures::ring_of_cliques;

    #[test]
    fn the_ladder_is_increasing_and_spans_the_whole_useful_range() {
        let g = ring_of_cliques(20, 5);
        let l = ladder(&g, 32);
        assert_eq!(l.len(), 32);
        assert!(
            l.windows(2).all(|w| w[1] > w[0]),
            "the ladder is not sorted"
        );
        let n = g.n as f64;
        assert!((l[0] - 1.0 / (n * n)).abs() < 1e-15, "coarse end {}", l[0]);
        assert!((l[31] - 1.0).abs() < 1e-12, "fine end {}", l[31]);
    }

    #[test]
    fn degenerate_graphs_do_not_produce_nonsense() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        assert_eq!(ladder(&g, 4), vec![1.0; 4]);
        let g = CsrGraph::from_edges(&[0], &[]);
        assert_eq!(ladder(&g, 3), vec![1.0; 3]);

        let g = ring_of_cliques(4, 4);
        assert_eq!(ladder(&g, 2), vec![1.0 / 256.0, 1.0]);
    }
}
