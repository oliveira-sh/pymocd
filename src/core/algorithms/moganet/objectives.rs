//! MOGA-Net bi-objective (Community Score, Community Fitness) and the
//! modularity used as decision rule.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;

use super::locus::Locus;

/// `(community_score, community_fitness)` on a decoded label array (Pizzuti
/// 2012, Sec. V-A; equations in `README.md`). **Both are maximized** -- a
/// minimizing dominance rule must be fed `(-CS, -CF)`. `M(S)` is the mean of
/// `mu_i^r` with no outer root. Sums run in ascending position and
/// community-id order, so the float result is reproducible.
pub fn community_objectives(locus: &Locus, labels: &[i32], r: f64, alpha: f64) -> (f64, f64) {
    let n_comms = labels.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut size = vec![0usize; n_comms];
    for &c in labels {
        size[c as usize] += 1;
    }

    let mut mu_pow_sum = vec![0.0f64; n_comms];
    let mut v_s = vec![0.0f64; n_comms];
    let mut cf_sum = vec![0.0f64; n_comms];
    for (p, &lab) in labels.iter().enumerate() {
        let c = lab as usize;
        let mut k_in = 0usize;
        for &q in &locus.neighbors[p] {
            if labels[q] == lab {
                k_in += 1;
            }
        }
        let k = k_in as f64;
        v_s[c] += k;
        let mu = k / size[c] as f64;
        mu_pow_sum[c] += mu.powf(r);
        let deg = locus.neighbors[p].len() as f64;
        if deg > 0.0 {
            cf_sum[c] += k / deg.powf(alpha);
        }
    }

    let mut cs = 0.0;
    let mut cf = 0.0;
    for c in 0..n_comms {
        if size[c] == 0 {
            continue;
        }
        cs += (mu_pow_sum[c] / size[c] as f64) * v_s[c];
        cf += cf_sum[c];
    }
    (cs, cf)
}

/// Newman modularity on a position-indexed label array; matches
/// `core::metrics::modularity`, with the same fixed summation order.
pub fn label_modularity(graph: &Graph, locus: &Locus, labels: &[i32]) -> f64 {
    let m = graph.num_edges() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let n_comms = labels.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut l_c = vec![0.0f64; n_comms];
    let mut d_c = vec![0.0f64; n_comms];
    for (p, &lab) in labels.iter().enumerate() {
        let c = lab as usize;
        d_c[c] += locus.neighbors[p].len() as f64;
        for &q in &locus.neighbors[p] {
            if p < q && labels[q] == lab {
                l_c[c] += 1.0;
            }
        }
    }
    (0..n_comms)
        .map(|c| l_c[c] / m - (d_c[c] / (2.0 * m)).powi(2))
        .sum()
}
