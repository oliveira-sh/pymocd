//! The MMCoMO search objectives and the Newman modularity used for selection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mmcomo::{Graph, Labels};

/// Per-community `|V_i|`, `L(V_i,V_i)` and degree sum, in first-seen-label order
/// over ascending node index. That order is what makes both objectives sum
/// their terms the same way on every run; every entry has at least one member,
/// so `size` is never zero.
struct CommunityStats {
    size: Vec<f64>,
    l_in: Vec<f64>,
    deg_sum: Vec<f64>,
}

fn community_stats(g: &Graph, labels: &Labels) -> CommunityStats {
    let mut compact: Vec<i32> = vec![-1; g.n];
    let mut size: Vec<f64> = Vec::new();
    let mut l_in: Vec<f64> = Vec::new();
    let mut deg_sum: Vec<f64> = Vec::new();

    for v in 0..g.n {
        let lab = labels[v] as usize;
        if compact[lab] < 0 {
            compact[lab] = size.len() as i32;
            size.push(0.0);
            l_in.push(0.0);
            deg_sum.push(0.0);
        }
        let c = compact[lab] as usize;
        size[c] += 1.0;
        deg_sum[c] += g.deg[v];
        let mut internal = 0.0;
        for &u in &g.adj[v] {
            if labels[u] == labels[v] {
                internal += 1.0;
            }
        }
        l_in[c] += internal;
    }

    CommunityStats {
        size,
        l_in,
        deg_sum,
    }
}

/// Eq.1 — KKM/RC bi-objective (Shi 2012, Gong), both minimized.
///
/// ```text
///   KKM = 2(n − k) − Σ_i L(V_i, V_i) / |V_i|
///   RC  =            Σ_i L(V_i, V̄_i) / |V_i|
/// ```
/// `L(V_i,V_i)` = twice the internal-edge count; cut = Σ deg − L(V_i,V_i).
pub fn kkm_rc(g: &Graph, labels: &Labels) -> (f64, f64) {
    let stats = community_stats(g, labels);
    let n = g.n as f64;
    let k = stats.size.len() as f64;

    let mut kkm_internal = 0.0;
    let mut rc = 0.0;
    for ((&sz, &l_in), &deg_sum) in stats
        .size
        .iter()
        .zip(stats.l_in.iter())
        .zip(stats.deg_sum.iter())
    {
        kkm_internal += l_in / sz;
        rc += (deg_sum - l_in) / sz;
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

/// Newman modularity Q (maximized — used for selection / local search).
pub fn modularity(g: &Graph, labels: &Labels) -> f64 {
    let m2 = g.m2;
    if m2 == 0.0 {
        return 0.0;
    }
    let m = m2 / 2.0;

    let stats = community_stats(g, labels);

    let mut q = 0.0;
    for (&l_in, &deg_sum) in stats.l_in.iter().zip(stats.deg_sum.iter()) {
        let lc = l_in / 2.0; // l_in counts each internal edge twice
        q += lc / m - (deg_sum / m2).powi(2);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mmcomo::fixtures::two_triangles;

    #[test]
    fn kkm_rc_exact_values_on_split() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        // each community: L_in=6, |V|=3, deg_sum=7
        // KKM = 2(6−2) − (6/3 + 6/3) = 4 ; RC = (1/3 + 1/3) = 2/3.
        let (kkm, rc) = kkm_rc(&g, &split);
        assert!((kkm - 4.0).abs() < 1e-9, "KKM={kkm}");
        assert!((rc - 2.0 / 3.0).abs() < 1e-9, "RC={rc}");
    }

    #[test]
    fn kkm_rc_degenerate_extremes() {
        let g = two_triangles();
        let one: Labels = vec![0, 0, 0, 0, 0, 0];
        let singletons: Labels = vec![0, 1, 2, 3, 4, 5];

        let (_kkm_one, rc_one) = kkm_rc(&g, &one);
        let (kkm_sing, _rc_sing) = kkm_rc(&g, &singletons);

        assert!(kkm_sing.abs() < 1e-9, "singletons KKM={kkm_sing}");
        assert!(rc_one.abs() < 1e-9, "one-community RC={rc_one}");

        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let (kkm_split, rc_split) = kkm_rc(&g, &split);
        let (kkm_one, _) = kkm_rc(&g, &one);
        let (_, rc_sing) = kkm_rc(&g, &singletons);
        assert!(
            kkm_split < kkm_one,
            "KKM split {kkm_split} !< one {kkm_one}"
        );
        assert!(
            rc_split < rc_sing,
            "RC split {rc_split} !< singletons {rc_sing}"
        );
    }

    #[test]
    fn modularity_exact_value_on_split() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        // m = 7. Each community: l_c=3, d_c=7. Q = 2*(3/7 − 0.25) = 0.357142857.
        let q = modularity(&g, &split);
        assert!((q - 0.357142857142857).abs() < 1e-9, "Q={q}");
    }

    #[test]
    fn modularity_split_beats_one_community() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let one: Labels = vec![0, 0, 0, 0, 0, 0];
        let q_one = modularity(&g, &one);
        assert!(q_one.abs() < 1e-9, "one-community Q={q_one}");
        assert!(modularity(&g, &split) > q_one);
    }
}
