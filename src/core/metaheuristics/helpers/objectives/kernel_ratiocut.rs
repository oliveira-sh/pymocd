//! Kernel-K-Means (KKM) and Ratio-Cut (RC) community objectives — the bi-objective
//! used by NSGA-III-KRM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021) and
//! by MODPSO. Both are **minimized**. For a partition `C = {V_1..V_k}` on `n`
//! nodes, with `L(V_a,V_b) = Σ_{i∈V_a, j∈V_b} A_ij`:
//! ```text
//!   KKM = 2(n − k) − Σ_i L(V_i, V_i)/|V_i|      (denser communities ⇒ lower KKM)
//!   RC  =            Σ_i L(V_i, V̄_i)/|V_i|      (fewer inter-links ⇒ lower RC)
//! ```
//! `L(V_i,V_i)` = twice the internal-edge count of `V_i` (each internal edge is
//! counted from both endpoints) = `Σ_{v∈V_i}` #neighbours of `v` inside `V_i`;
//! `L(V_i,V̄_i)` = the cut = `Σ_{v∈V_i} deg(v) − L(V_i,V_i)`. KKM is minimized by
//! fragmentation (singletons → 0), RC by coarsening (one community → 0); the
//! good structure lives on the Pareto trade-off between them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rustc_hash::FxHashMap;

/// Returns `(KKM, RC)`, both **minimized**. Mirrors
/// `objectives::decomposed_modularity::calculate_objectives`: build communities
/// from `partition`, accumulate internal degree (`L_in`) via `graph.neighbors`
/// and total degree via `graph.degree`.
pub fn kkm_ratiocut(graph: &Graph, partition: &Partition) -> (f64, f64) {
    let mut comms: FxHashMap<CommunityId, Vec<NodeId>> = FxHashMap::default();
    for (&node, &c) in partition.iter() {
        comms.entry(c).or_default().push(node);
    }

    let n = partition.len() as f64;
    let k = comms.len() as f64;

    let mut kkm_internal = 0.0; // Σ L(V_i,V_i)/|V_i|
    let mut rc = 0.0; // Σ L(V_i,V̄_i)/|V_i|
    for (&c, nodes) in comms.iter() {
        let size = nodes.len() as f64;
        if size == 0.0 {
            continue;
        }
        let mut l_in = 0.0; // = 2·internal edges
        let mut deg_sum = 0.0;
        for &v in nodes {
            deg_sum += graph.degree(&v) as f64;
            for &u in graph.neighbors(&v) {
                if partition.get(&u) == Some(&c) {
                    l_in += 1.0;
                }
            }
        }
        kkm_internal += l_in / size;
        rc += (deg_sum - l_in) / size; // cut = total degree − internal degree
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{CommunityId, NodeId};

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    fn part(pairs: &[(NodeId, CommunityId)]) -> Partition {
        pairs.iter().copied().collect()
    }

    #[test]
    fn split_sits_between_the_degenerate_extremes() {
        let g = two_triangles();
        let split = part(&[(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]);
        let one = part(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
        let singletons = part(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);

        let (kkm_split, rc_split) = kkm_ratiocut(&g, &split);
        let (kkm_one, rc_one) = kkm_ratiocut(&g, &one);
        let (kkm_sing, rc_sing) = kkm_ratiocut(&g, &singletons);
        assert_eq!(kkm_sing, 0.0, "singletons are KKM's fragmentation extreme");

        // Exact values for the good split: each community has L_in=6, |V|=3,
        // deg_sum=7 → KKM = 2(6−2) − (6/3+6/3) = 4 ; RC = (1/3 + 1/3) = 2/3.
        assert!((kkm_split - 4.0).abs() < 1e-9, "KKM={kkm_split}");
        assert!((rc_split - 2.0 / 3.0).abs() < 1e-9, "RC={rc_split}");

        // KKM penalizes under-segmentation: the split beats the all-in-one blob.
        assert!(kkm_split < kkm_one, "KKM split {kkm_split} !< one {kkm_one}");
        // RC penalizes over-segmentation: the split beats the singletons. (The
        // all-in-one blob has RC=0 — zero cut — so RC alone cannot rule it out;
        // that is exactly why KKM is the second, opposing objective.)
        assert!(rc_split < rc_sing, "RC split {rc_split} !< singletons {rc_sing}");
        assert_eq!(rc_one, 0.0, "all-in-one has no cut");
    }
}
