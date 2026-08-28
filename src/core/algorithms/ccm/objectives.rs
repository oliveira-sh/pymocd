//! CCM's three objectives on decoded label arrays: MOGA-Net's (Community Score,
//! Community Fitness) plus Newman modularity.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// The paper maximizes all three; `Individual::dominates` assumes
/// minimization, so what the engine gets is `(-CS, -CF, -Q)`.
pub fn evaluate(neighbor_pos: &[Vec<usize>], labels: &[i32], r: f64, alpha: f64) -> Vec<f64> {
    let (cs, cf) = community_objectives(neighbor_pos, labels, r, alpha);
    let q = modularity_labels(neighbor_pos, labels);
    vec![-cs, -cf, -q]
}

/// MOGA-Net's bi-objective (Pizzuti, IEEE TEC 16(3):418–430, 2012, Sec. V-A),
/// **both maximized**. For node `i` of community `S`, `k_in` = its neighbours
/// inside `S`:
/// ```text
///   mu_i     = k_in / |S|                       (|S| = node count)
///   M(S)     = (Σ_{i∈S} mu_i^r) / |S|           (mean of mu_i^r — NO outer root)
///   v_S      = Σ_{i∈S} k_in = 2·(internal edges of S)
///   score(S) = M(S) · v_S ;  CS = Σ_S score(S)
///   CF       = Σ_S Σ_{i∈S} k_in / deg(i)^α      (deg(i)=0 → term 0)
/// ```
pub fn community_objectives(
    neighbor_pos: &[Vec<usize>],
    labels: &[i32],
    r: f64,
    alpha: f64,
) -> (f64, f64) {
    let n = labels.len();
    let mut size = vec![0.0f64; n];
    for &c in labels {
        size[c as usize] += 1.0;
    }

    let mut mu_pow_sum = vec![0.0f64; n];
    let mut internal_degree_sum = vec![0.0f64; n];
    let mut fitness_sum = vec![0.0f64; n];
    for p in 0..n {
        let c = labels[p] as usize;
        let k_in = neighbor_pos[p]
            .iter()
            .filter(|&&q| labels[q] == labels[p])
            .count();
        let k = k_in as f64;
        internal_degree_sum[c] += k;
        let mu = k / size[c];
        mu_pow_sum[c] += mu.powf(r);
        let deg = neighbor_pos[p].len() as f64;
        if deg > 0.0 {
            fitness_sum[c] += k / deg.powf(alpha);
        }
    }

    let mut cs = 0.0;
    let mut cf = 0.0;
    for c in 0..n {
        if size[c] > 0.0 {
            cs += (mu_pow_sum[c] / size[c]) * internal_degree_sum[c];
            cf += fitness_sum[c];
        }
    }
    (cs, cf)
}

/// Newman modularity `Q = Σ_c l_c/m − (d_c/2m)²` (internal edges `l_c` counted
/// once, `d_c` = Σ degrees), kept local to `core::metrics::modularity` so the
/// hot loop never builds a `Partition`.
pub fn modularity_labels(neighbor_pos: &[Vec<usize>], labels: &[i32]) -> f64 {
    let n = labels.len();
    let two_m: usize = neighbor_pos.iter().map(std::vec::Vec::len).sum();
    let m = two_m as f64 / 2.0;
    if m == 0.0 {
        return 0.0;
    }

    let mut l_c = vec![0.0f64; n];
    let mut d_c = vec![0.0f64; n];
    for p in 0..n {
        let c = labels[p] as usize;
        d_c[c] += neighbor_pos[p].len() as f64;
        for &q in &neighbor_pos[p] {
            if p < q && labels[q] == labels[p] {
                l_c[c] += 1.0;
            }
        }
    }

    let mut q_sum = 0.0;
    for c in 0..n {
        if d_c[c] > 0.0 || l_c[c] > 0.0 {
            q_sum += l_c[c] / m - (d_c[c] / (2.0 * m)).powi(2);
        }
    }
    q_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ccm::fixtures::two_triangles;
    use crate::core::algorithms::ccm::locus;
    use crate::core::graph::Partition;
    use crate::core::metrics::modularity::modularity;

    #[test]
    fn modularity_labels_matches_shared_metric() {
        let g = two_triangles();
        let nodes = g.nodes_vec().clone();
        let neighbor_pos = locus::neighbor_positions(&g, &nodes);
        let labels: Vec<i32> = vec![0, 0, 0, 3, 3, 3];
        let part: Partition = nodes
            .iter()
            .enumerate()
            .map(|(p, &node)| (node, labels[p]))
            .collect();
        let q = modularity_labels(&neighbor_pos, &labels);
        assert!((q - modularity(&g, &part)).abs() < 1e-12);
    }
}
