//! Pizzuti's MOGA-Net bi-objective (ICTAI 2009 / IEEE TEC 16(3):418–430, 2012)
//! plus Newman modularity, all evaluated on decoded label arrays.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// MOGA-Net bi-objective on decoded labels (Pizzuti 2012, Sec. V-A).
/// `labels[p]` is the community of position `p` (union-find root positions in
/// `0..n`, so accumulators index them directly). Returns
/// `(community_score, community_fitness)` = `(CS, CF)`, **both maximized**
/// (Pizzuti maximizes CS and the community fitness — the latter peaks when no
/// edges leave a community, i.e. maximizing it minimizes inter-module links). For
/// node `i` in its community `S`, with `k_in` = neighbours of `i` inside `S`:
/// ```text
///   mu_i   = k_in / |S|                         (|S| = node count, ∈ [0,1))
///   M(S)   = (Σ_{i∈S} mu_i^r) / |S|             (mean of mu_i^r — NO outer root)
///   v_S    = Σ_{i∈S} k_in = 2·(internal edges of S)
///   score(S) = M(S) · v_S ; CS = Σ_S score(S)
///   CF     = Σ_S Σ_{i∈S} k_in / deg(i)^α        (deg(i)=0 → term 0)
/// ```
/// An NSGA-II that minimizes feeds `(−CS, −CF)`.
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

    let mut m_num = vec![0.0f64; n]; // per community: Σ mu_i^r
    let mut v_s = vec![0.0f64; n]; // per community: Σ k_in = 2·internal edges
    let mut p_s = vec![0.0f64; n]; // per community: Σ k_in / deg^α
    for p in 0..n {
        let c = labels[p] as usize;
        let k_in = neighbor_pos[p]
            .iter()
            .filter(|&&q| labels[q] == labels[p])
            .count();
        let k = k_in as f64;
        v_s[c] += k;
        let mu = k / size[c];
        m_num[c] += mu.powf(r);
        let deg = neighbor_pos[p].len() as f64;
        if deg > 0.0 {
            p_s[c] += k / deg.powf(alpha);
        }
    }

    let mut cs = 0.0;
    let mut cf = 0.0;
    for c in 0..n {
        if size[c] > 0.0 {
            cs += (m_num[c] / size[c]) * v_s[c];
            cf += p_s[c];
        }
    }
    (cs, cf)
}

/// Newman modularity `Q = Σ_c l_c/m − (d_c/2m)²` on decoded labels (internal
/// edges `l_c` counted once). Local so the hot loop never builds a `Partition`.
pub fn modularity_labels(neighbor_pos: &[Vec<usize>], labels: &[i32]) -> f64 {
    let n = labels.len();
    let two_m: usize = neighbor_pos.iter().map(|nb| nb.len()).sum();
    let m = two_m as f64 / 2.0;
    if m == 0.0 {
        return 0.0;
    }

    let mut l_c = vec![0.0f64; n]; // per community: internal edges, counted once
    let mut d_c = vec![0.0f64; n]; // per community: Σ degrees
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
    use crate::core::algorithms::ccm::locus;
    use crate::core::graph::{Graph, Partition};
    use crate::core::metrics::modularity::modularity;

    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

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
