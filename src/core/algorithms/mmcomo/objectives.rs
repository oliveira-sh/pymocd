use super::*;

use std::collections::HashMap;

/// Eq.1 — KKM/RC bi-objective (Shi 2012, Gong), both minimized.
///
/// ```text
///   KKM = 2(n − k) − Σ_i L(V_i, V_i) / |V_i|
///   RC  =            Σ_i L(V_i, V̄_i) / |V_i|
/// ```
/// `L(V_i,V_i)` = twice the internal-edge count; cut = Σ deg − L(V_i,V_i).
pub fn kkm_rc(g: &Graph, labels: &Labels) -> (f64, f64) {
    let mut size: HashMap<i32, f64> = HashMap::new();
    let mut l_in: HashMap<i32, f64> = HashMap::new();
    let mut deg_sum: HashMap<i32, f64> = HashMap::new();

    for v in 0..g.n {
        let c = labels[v];
        *size.entry(c).or_insert(0.0) += 1.0;
        *deg_sum.entry(c).or_insert(0.0) += g.deg[v];
        let mut internal = 0.0;
        for &u in &g.adj[v] {
            if labels[u] == c {
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

/// Newman modularity Q (maximized — used for selection / local search).
pub fn modularity(g: &Graph, labels: &Labels) -> f64 {
    let m2 = g.m2;
    if m2 == 0.0 {
        return 0.0;
    }
    let m = m2 / 2.0;

    let mut l_in: HashMap<i32, f64> = HashMap::new();
    let mut deg_sum: HashMap<i32, f64> = HashMap::new();

    for v in 0..g.n {
        let c = labels[v];
        *deg_sum.entry(c).or_insert(0.0) += g.deg[v];
        let mut internal = 0.0;
        for &u in &g.adj[v] {
            if labels[u] == c {
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

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        let n = 6;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2: f64 = deg.iter().sum();
        Graph { n, adj, deg, m2 }
    }

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
        assert!(kkm_split < kkm_one, "KKM split {kkm_split} !< one {kkm_one}");
        assert!(rc_split < rc_sing, "RC split {rc_split} !< singletons {rc_sing}");
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
