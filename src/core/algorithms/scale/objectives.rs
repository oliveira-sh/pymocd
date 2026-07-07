//! Objectives for `scale`, ported to `CsrGraph` (flat CSR, FxHashMap by label).

use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;

use super::Labels;

/// Eq.1 — KKM/RC bi-objective (Shi 2012, Gong), both minimized.
///
/// ```text
///   KKM = 2(n − k) − Σ_i L(V_i, V_i) / |V_i|
///   RC  =            Σ_i L(V_i, V̄_i) / |V_i|
/// ```
/// `L(V_i,V_i)` = twice the internal-edge count; cut = Σ deg − L(V_i,V_i).
pub fn kkm_rc(g: &CsrGraph, labels: &Labels) -> (f64, f64) {
    let mut size: FxHashMap<i32, f64> = FxHashMap::default();
    let mut l_in: FxHashMap<i32, f64> = FxHashMap::default();
    let mut deg_sum: FxHashMap<i32, f64> = FxHashMap::default();

    for v in 0..g.n {
        let c = labels[v];
        *size.entry(c).or_insert(0.0) += 1.0;
        *deg_sum.entry(c).or_insert(0.0) += g.deg[v] as f64;
        let mut internal = 0.0;
        for &u in g.neighbors(v) {
            if labels[u as usize] == c {
                internal += 1.0;
            }
        }
        *l_in.entry(c).or_insert(0.0) += internal;
    }

    let n = g.n as f64;
    let k = size.len() as f64;

    let mut kkm_internal = 0.0;
    let mut rc = 0.0;
    for (c, &sz) in size.iter() {
        if sz == 0.0 {
            continue;
        }
        let li = l_in[c];
        let ds = deg_sum[c];
        kkm_internal += li / sz;
        rc += (ds - li) / sz;
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

/// Newman modularity Q (maximized). The max-Q selector alternative (runner-up to
/// SBM/MDL in the Phase 4 oracle-gap study); retained for reference and tests.
#[allow(dead_code)]
pub fn modularity(g: &CsrGraph, labels: &Labels) -> f64 {
    let m2 = (2 * g.m) as f64;
    if m2 == 0.0 {
        return 0.0;
    }
    let m = m2 / 2.0;

    let mut l_in: FxHashMap<i32, f64> = FxHashMap::default();
    let mut deg_sum: FxHashMap<i32, f64> = FxHashMap::default();

    for v in 0..g.n {
        let c = labels[v];
        *deg_sum.entry(c).or_insert(0.0) += g.deg[v] as f64;
        let mut internal = 0.0;
        for &u in g.neighbors(v) {
            if labels[u as usize] == c {
                internal += 1.0;
            }
        }
        *l_in.entry(c).or_insert(0.0) += internal;
    }

    let mut q = 0.0;
    for (c, &ds) in deg_sum.iter() {
        let lc = l_in[c] / 2.0; // l_in counts each internal edge twice
        q += lc / m - (ds / m2).powi(2);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::CsrGraph;

    fn two_triangles() -> CsrGraph {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        CsrGraph::from_edges(&nodes, &edges)
    }

    #[test]
    fn kkm_rc_exact_values_on_split() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let (kkm, rc) = kkm_rc(&g, &split);
        assert!((kkm - 4.0).abs() < 1e-9, "KKM={kkm}");
        assert!((rc - 2.0 / 3.0).abs() < 1e-9, "RC={rc}");
    }

    #[test]
    fn modularity_exact_value_on_split() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let q = modularity(&g, &split);
        assert!((q - 0.357142857142857).abs() < 1e-9, "Q={q}");
    }

    #[test]
    fn modularity_split_beats_one_community() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let one: Labels = vec![0, 0, 0, 0, 0, 0];
        assert!(modularity(&g, &split) > modularity(&g, &one));
    }
}
