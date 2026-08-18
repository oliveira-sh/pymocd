//! The resolution ladder: each particle's niche on the CPM resolution profile.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

/// The resolution assigned to each of `pop` particles, spaced geometrically over
/// `[1/n^2, 1]` — the whole range where `gamma` can still change the answer, since a CPM
/// community must be internally denser than `gamma` to survive.
pub fn ladder(g: &CsrGraph, pop: usize) -> Vec<f64> {
    let n = g.n;
    if n < 2 || g.m == 0 {
        return vec![1.0; pop];
    }
    let n = n as f64;
    (0..pop)
        .map(|k| {
            let t = if pop > 1 {
                k as f64 / (pop - 1) as f64
            } else {
                0.5
            };
            n.powf(2.0 * t - 2.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::utils::fixtures::ring_of_cliques;

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
    fn the_ends_are_set_by_cpm_and_not_by_the_graph() {
        for (k, s) in [(4, 5), (40, 5), (10, 30)] {
            let g = ring_of_cliques(k, s);
            let l = ladder(&g, 8);
            let n = g.n as f64;
            assert!(
                (l[0] * n * n - 1.0).abs() < 1e-9,
                "n={n}: coarse end {}",
                l[0]
            );
            assert!((l[7] - 1.0).abs() < 1e-12, "n={n}: fine end {}", l[7]);
        }
    }

    #[test]
    fn the_ladder_brackets_the_planted_granularity() {
        // Merging two adjacent 5-cliques gains one ring edge for a penalty of 25 gamma,
        // so the planted partition is the CPM optimum exactly on 0.04 < gamma < 1.
        let g = ring_of_cliques(30, 5);
        let l = ladder(&g, 100);
        assert!(
            l.iter().filter(|&&x| x > 0.04 && x < 1.0).count() >= 8,
            "too few rungs inside the planted plateau: {l:?}"
        );
    }

    #[test]
    fn degenerate_graphs_do_not_produce_nonsense() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        assert_eq!(ladder(&g, 4), vec![1.0; 4]);
        let g = CsrGraph::from_edges(&[0], &[]);
        assert_eq!(ladder(&g, 3), vec![1.0; 3]);

        let g = ring_of_cliques(4, 4);
        assert_eq!(ladder(&g, 1).len(), 1);
        assert!(ladder(&g, 1)[0].is_finite() && ladder(&g, 1)[0] > 0.0);
    }
}
