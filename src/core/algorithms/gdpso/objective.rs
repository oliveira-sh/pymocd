//! GDPSO's single maximised objective: Newman-Girvan modularity, plus its gain.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

/// Degree sum of every community, indexed by label; `sigma` must be `n` long.
pub fn degree_sums(g: &CsrGraph, labels: &[i32], sigma: &mut [i64]) {
    sigma.fill(0);
    for (u, &lab) in labels.iter().enumerate() {
        sigma[lab as usize] += i64::from(g.deg[u]);
    }
}

/// Newman-Girvan modularity `Q = sum_c [ e_c/m - (K_c/2m)^2 ]`, maximised,
/// with `e_c` counting each internal undirected edge once and `K_c = sigma[c]`.
/// An edgeless graph is defined to have `Q = 0` rather than dividing by zero.
pub fn modularity(g: &CsrGraph, labels: &[i32], sigma: &[i64]) -> f64 {
    if g.m == 0 {
        return 0.0;
    }
    let m = g.m as f64;
    let mut intra = 0usize;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            intra += 1;
        }
    }
    // fixed summation order over the label index, so Q is bit-reproducible
    let mut null = 0.0;
    for &k in sigma {
        let share = k as f64 / (2.0 * m);
        null += share * share;
    }
    intra as f64 / m - null
}

/// The comparable half of the modularity gain for putting node `i` in a
/// community: `kappa - k_i * sigma / 2m`, where `kappa` counts `i`'s edges into
/// the community and `sigma` is that community's degree sum **excluding** `i`.
///
/// `delta_q(i -> l) = (score(l) - score(own)) / m`, so ranking by this score and
/// accepting only a strict increase is exactly best improvement with a strictly
/// positive gain.
#[inline]
pub fn move_score(kappa: f64, k_i: f64, sigma: f64, inv_two_m: f64) -> f64 {
    kappa - k_i * sigma * inv_two_m
}

/// Exact single-node modularity gain, the identity `move_score` is derived from.
#[cfg(test)]
pub fn delta_q(g: &CsrGraph, labels: &[i32], sigma: &[i64], i: usize, to: i32) -> f64 {
    let m = g.m as f64;
    let k_i = f64::from(g.deg[i]);
    let own = labels[i];
    let (mut kappa_to, mut kappa_own) = (0.0, 0.0);
    for &q in g.neighbors(i) {
        if labels[q as usize] == to {
            kappa_to += 1.0;
        } else if labels[q as usize] == own {
            kappa_own += 1.0;
        }
    }
    let sigma_to = sigma[to as usize] as f64;
    let sigma_own = sigma[own as usize] as f64 - k_i;
    (kappa_to - kappa_own) / m - k_i * (sigma_to - sigma_own) / (2.0 * m * m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> CsrGraph {
        CsrGraph::from_edges(
            &(0..6).collect::<Vec<i32>>(),
            &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)],
        )
    }

    fn q_of(g: &CsrGraph, labels: &[i32]) -> f64 {
        let mut sigma = vec![0i64; g.n];
        degree_sums(g, labels, &mut sigma);
        modularity(g, labels, &sigma)
    }

    #[test]
    fn modularity_matches_the_closed_form() {
        let g = two_triangles();
        // m = 7, each side has 3 internal edges and degree sum 7
        let split = [0, 0, 0, 1, 1, 1];
        let expected = 6.0 / 7.0 - 2.0 * (7.0 / 14.0f64).powi(2);
        assert!((q_of(&g, &split) - expected).abs() < 1e-12);
        // one community: every edge internal against the whole null model
        assert!(q_of(&g, &[0, 0, 0, 0, 0, 0]).abs() < 1e-12);
        assert!(q_of(&g, &[0, 1, 2, 3, 4, 5]) < 0.0);
    }

    #[test]
    fn delta_q_equals_the_recomputed_difference() {
        let g = two_triangles();
        let labels = [0, 0, 0, 1, 1, 1];
        let mut sigma = vec![0i64; g.n];
        degree_sums(&g, &labels, &mut sigma);
        for i in 0..g.n {
            for to in 0..2i32 {
                if labels[i] == to {
                    continue;
                }
                let mut moved = labels;
                moved[i] = to;
                let exact = q_of(&g, &moved) - q_of(&g, &labels);
                let gain = delta_q(&g, &labels, &sigma, i, to);
                assert!((gain - exact).abs() < 1e-12, "node {i} -> {to}");
            }
        }
    }

    #[test]
    fn move_score_ranks_the_same_way_as_delta_q() {
        let g = two_triangles();
        let labels = [0, 0, 0, 1, 1, 1];
        let mut sigma = vec![0i64; g.n];
        degree_sums(&g, &labels, &mut sigma);
        let inv_two_m = 1.0 / (2.0 * g.m as f64);
        // node 2 sits on the bridge: 2 neighbours in its own community, 1 across
        let k_i = f64::from(g.deg[2]);
        let own = move_score(2.0, k_i, sigma[0] as f64 - k_i, inv_two_m);
        let other = move_score(1.0, k_i, sigma[1] as f64, inv_two_m);
        let gain = delta_q(&g, &labels, &sigma, 2, 1);
        assert!((other - own) / g.m as f64 - gain < 1e-12);
        assert!(other < own, "the bridge node must stay put");
    }

    #[test]
    fn empty_graph_has_zero_modularity() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        assert!(q_of(&g, &[0, 1, 2]).abs() < 1e-12);
    }
}
