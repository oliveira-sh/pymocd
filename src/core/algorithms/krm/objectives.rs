//! The (KKM, RC) search objectives and the modularity that picks from the front.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;

use super::locus::Locus;

/// Per-community `(l_in, deg_sum, size)`, indexed by community id compacted in
/// ascending position order — a fixed summation order. `l_in` is `L(V_i,V_i)`:
/// each internal edge counted from both endpoints, so twice the edge count.
fn community_stats(locus: &Locus, labels: &[i32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = labels.len();
    let mut compact = vec![-1i32; n];
    let mut l_in: Vec<f64> = Vec::new();
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
        let cand = &locus.candidates[p];
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

/// Kernel-K-Means and Ratio-Cut, two of KRM's three objectives, both minimized:
/// ```text
///   KKM = 2(n − k) − Σ_i L(V_i, V_i)/|V_i|
///   RC  =            Σ_i L(V_i, V̄_i)/|V_i|
/// ```
pub fn kkm_ratiocut(locus: &Locus, labels: &[i32]) -> (f64, f64) {
    let (l_in, deg_sum, size) = community_stats(locus, labels);
    let n = labels.len() as f64;
    let k = size.len() as f64;

    let mut kkm_internal = 0.0;
    let mut rc = 0.0;
    for c in 0..size.len() {
        kkm_internal += l_in[c] / size[c];
        rc += (deg_sum[c] - l_in[c]) / size[c];
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

/// Newman `Q` on a label array. Duplicates `metrics::modularity` on purpose:
/// the loop never builds a shared `Partition`.
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
    use crate::core::algorithms::krm::fixtures::two_triangles;

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

        // Each community has L_in=6, |V|=3, deg_sum=7 → KKM = 2(6−2) −
        // (6/3+6/3) = 4 ; RC = (1/3 + 1/3) = 2/3.
        assert!((kkm_split - 4.0).abs() < 1e-9, "KKM={kkm_split}");
        assert!((rc_split - 2.0 / 3.0).abs() < 1e-9, "RC={rc_split}");

        assert!(
            kkm_split < kkm_one,
            "KKM split {kkm_split} !< one {kkm_one}"
        );
        assert!(
            rc_split < rc_sing,
            "RC split {rc_split} !< singletons {rc_sing}"
        );
        assert_eq!(rc_one, 0.0, "all-in-one has no cut");

        // m=7, each community l=3, d=7 → Q = 2(3/7 − (7/14)²).
        let q = modularity(&g, &locus, &split);
        assert!((q - (6.0 / 7.0 - 0.5)).abs() < 1e-12, "Q={q}");
    }
}
