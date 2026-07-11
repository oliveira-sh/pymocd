//! Pizzuti's MOGA-Net bi-objective (ICTAI 2009 / IEEE TEC 16(3):418–430, 2012).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rustc_hash::FxHashMap;

/// MOGA-Net bi-objective on a decoded label partition (Pizzuti 2012, Sec. V-A).
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
/// An NSGA-II that minimizes feeds `(−CS, −CF)`.
pub fn community_objectives(
    graph: &Graph,
    partition: &Partition,
    r: f64,
    alpha: f64,
) -> (f64, f64) {
    let mut comms: FxHashMap<CommunityId, Vec<NodeId>> = FxHashMap::default();
    for (&node, &c) in partition.iter() {
        comms.entry(c).or_default().push(node);
    }

    let mut cs = 0.0;
    let mut cf = 0.0;
    for (&c, nodes) in comms.iter() {
        let s_size = nodes.len() as f64;
        let mut m_num = 0.0; // Σ mu_i^r
        let mut v_s = 0.0; // Σ k_in = 2·internal edges
        let mut p_s = 0.0; // Σ k_in / deg^α
        for &i in nodes {
            let mut k_in = 0usize;
            for &j in graph.neighbors(&i) {
                if partition.get(&j) == Some(&c) {
                    k_in += 1;
                }
            }
            let k = k_in as f64;
            v_s += k;
            let mu = k / s_size;
            m_num += mu.powf(r);
            let deg = graph.degree(&i) as f64;
            if deg > 0.0 {
                p_s += k / deg.powf(alpha);
            }
        }
        cs += (m_num / s_size) * v_s;
        cf += p_s;
    }
    (cs, cf)
}
