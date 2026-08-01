//! Kernel-K-Means (KKM) and Ratio-Cut (RC) community objectives — the bi-objective
//! used by NSGA-III-KRM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021) and
//! by MODPSO — plus in-module Newman modularity Q, all evaluated directly on
//! per-position label arrays. KKM and RC are **minimized**. For a partition
//! `C = {V_1..V_k}` on `n` nodes, with `L(V_a,V_b) = Σ_{i∈V_a, j∈V_b} A_ij`:
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

use super::locus::Locus;
use crate::core::graph::Graph;

/// Per-community accumulators over `labels` (raw ids must be `< n`, as decode
/// guarantees): `L(V_i,V_i)`, total degree, and size, indexed by community id
/// compacted in ascending position order (deterministic summation order).
fn community_stats(locus: &Locus, labels: &[i32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = labels.len();
    let mut compact = vec![-1i32; n];
    let mut l_in: Vec<f64> = Vec::new(); // Σ_{v∈V_i} #neighbours inside = 2·internal edges
    let mut deg_sum: Vec<f64> = Vec::new();
    let mut size: Vec<f64> = Vec::new();
    for p in 0..n {
        let raw = labels[p] as usize;
        if compact[raw] < 0 {
            compact[raw] = l_in.len() as i32;
            l_in.push(0.0);
            deg_sum.push(0.0);
            size.push(0.0);
        }
        let c = compact[raw] as usize;
        let cand = &locus.candidates[p]; // [p, neighbour positions...]
        size[c] += 1.0;
        deg_sum[c] += (cand.len() - 1) as f64;
        for &q in &cand[1..] {
            if labels[q] == labels[p] {
                l_in[c] += 1.0;
            }
        }
    }
    (l_in, deg_sum, size)
}

/// Returns `(KKM, RC)`, both **minimized**, from the per-community stats:
/// `KKM = 2(n−k) − Σ L_in/|V_i|`, `RC = Σ (deg_sum − L_in)/|V_i|` (the cut).
pub fn kkm_ratiocut(locus: &Locus, labels: &[i32]) -> (f64, f64) {
    let (l_in, deg_sum, size) = community_stats(locus, labels);
    let n = labels.len() as f64;
    let k = size.len() as f64;

    let mut kkm_internal = 0.0; // Σ L(V_i,V_i)/|V_i|
    let mut rc = 0.0; // Σ L(V_i,V̄_i)/|V_i|
    for c in 0..size.len() {
        kkm_internal += l_in[c] / size[c];
        rc += (deg_sum[c] - l_in[c]) / size[c];
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

/// Newman modularity `Q = Σ_c l_c/m − (d_c/2m)²` on a label array — same
/// formula as `metrics::modularity`, kept in-module so the evolutionary loop
/// never builds a shared `Partition`.
pub fn modularity(graph: &Graph, locus: &Locus, labels: &[i32]) -> f64 {
    let m = graph.num_edges() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let (l_in, deg_sum, _size) = community_stats(locus, labels);
    let mut q = 0.0;
    for c in 0..l_in.len() {
        q += l_in[c] / (2.0 * m) - (deg_sum[c] / (2.0 * m)).powi(2);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn split_sits_between_the_degenerate_extremes() {
        let g = two_triangles();
        let locus = Locus::build(&g); // sorted nodes 0..5 → position == node id
        let split = vec![0, 0, 0, 1, 1, 1];
        let one = vec![0, 0, 0, 0, 0, 0];
        let singletons = vec![0, 1, 2, 3, 4, 5];

        let (kkm_split, rc_split) = kkm_ratiocut(&locus, &split);
        let (kkm_one, rc_one) = kkm_ratiocut(&locus, &one);
        let (kkm_sing, rc_sing) = kkm_ratiocut(&locus, &singletons);
        assert_eq!(kkm_sing, 0.0, "singletons are KKM's fragmentation extreme");

        // Exact values for the good split: each community has L_in=6, |V|=3,
        // deg_sum=7 → KKM = 2(6−2) − (6/3+6/3) = 4 ; RC = (1/3 + 1/3) = 2/3.
        assert!((kkm_split - 4.0).abs() < 1e-9, "KKM={kkm_split}");
        assert!((rc_split - 2.0 / 3.0).abs() < 1e-9, "RC={rc_split}");

        // KKM penalizes under-segmentation: the split beats the all-in-one blob.
        assert!(
            kkm_split < kkm_one,
            "KKM split {kkm_split} !< one {kkm_one}"
        );
        // RC penalizes over-segmentation: the split beats the singletons. (The
        // all-in-one blob has RC=0 — zero cut — so RC alone cannot rule it out;
        // that is exactly why KKM is the second, opposing objective.)
        assert!(
            rc_split < rc_sing,
            "RC split {rc_split} !< singletons {rc_sing}"
        );
        assert_eq!(rc_one, 0.0, "all-in-one has no cut");

        // Q for the split: m=7, each community l=3, d=7 → 2(3/7 − (7/14)²).
        let q = modularity(&g, &locus, &split);
        assert!((q - (6.0 / 7.0 - 0.5)).abs() < 1e-12, "Q={q}");
    }
}
