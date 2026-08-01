//! Pizzuti's MOGA-Net bi-objective (ICTAI 2009 / IEEE TEC 16(3):418–430, 2012).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::Locus;

/// MOGA-Net bi-objective on a decoded label array (Pizzuti 2012, Sec. V-A).
/// Returns `(community_score, community_fitness)` = `(CS, CF)`, **both maximized**
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
/// An NSGA-II that minimizes feeds `(−CS, −CF)`. `labels` is indexed by
/// position with compact community ids; sums run in ascending position /
/// community-id order so floating-point summation is deterministic.
pub fn community_objectives(locus: &Locus, labels: &[i32], r: f64, alpha: f64) -> (f64, f64) {
    let n_comms = labels.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut size = vec![0usize; n_comms];
    for &c in labels {
        size[c as usize] += 1;
    }

    let mut m_num = vec![0.0f64; n_comms]; // Σ mu_i^r
    let mut v_s = vec![0.0f64; n_comms]; // Σ k_in = 2·internal edges
    let mut p_s = vec![0.0f64; n_comms]; // Σ k_in / deg^α
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
        m_num[c] += mu.powf(r);
        let deg = locus.neighbors[p].len() as f64;
        if deg > 0.0 {
            p_s[c] += k / deg.powf(alpha);
        }
    }

    let mut cs = 0.0;
    let mut cf = 0.0;
    for c in 0..n_comms {
        if size[c] == 0 {
            continue;
        }
        cs += (m_num[c] / size[c] as f64) * v_s[c];
        cf += p_s[c];
    }
    (cs, cf)
}
