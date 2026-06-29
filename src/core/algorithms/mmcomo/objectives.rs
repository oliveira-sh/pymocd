use super::*;

use std::collections::HashMap;

/// Eq.1 — the MODPSO / RMOEA (Shi 2012, Gong) bi-objective, BOTH MINIMIZED.
///
/// For a partition `C = {V_1,…,V_k}` on `n` nodes:
/// ```text
///   KKM = 2(n − k) − Σ_i L(V_i, V_i) / |V_i|
///   RC  =            Σ_i L(V_i, V̄_i) / |V_i|
/// ```
/// with `L(V_i,V_i)` = Σ_{v∈V_i} (#neighbours of v inside V_i) = twice the
/// internal-edge count, and the cut `L(V_i,V̄_i)` = Σ_{v∈V_i} deg(v) − L(V_i,V_i).
/// `k` = number of distinct labels.
///
/// KKM → 0 under fragmentation (singletons), RC → 0 under coarsening (one
/// community); the good structure lives on the Pareto trade-off between them.
pub fn kkm_rc(g: &Graph, labels: &Labels) -> (f64, f64) {
    // Per-community accumulators: (|V_i|, L(V_i,V_i), Σ deg).
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

    let mut kkm_internal = 0.0; // Σ L(V_i,V_i)/|V_i|
    let mut rc = 0.0; // Σ L(V_i,V̄_i)/|V_i|
    for (c, &sz) in size.iter() {
        if sz == 0.0 {
            continue;
        }
        let li = l_in[c];
        let ds = deg_sum[c];
        kkm_internal += li / sz;
        rc += (ds - li) / sz; // cut = total degree − internal degree
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

/// Newman modularity Q (to be MAXIMIZED — used for selection / local search).
///
/// `Q = Σ_c [ l_c/m − (d_c/2m)^2 ]`, where `m = |E| = m2/2`, `l_c` = internal
/// edge count of community `c`, `d_c` = total degree of community `c`.
pub fn modularity(g: &Graph, labels: &Labels) -> f64 {
    let m2 = g.m2;
    if m2 == 0.0 {
        return 0.0;
    }
    let m = m2 / 2.0;

    // l_in here = 2 × (internal edges); d_c = degree sum.
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
        let lc = l_in[c] / 2.0; // internal edges (l_in counts each twice)
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
        // good split: two triangles
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        // each community: L_in=6, |V|=3, deg_sum=7
        // KKM = 2(6−2) − (6/3 + 6/3) = 8 − 4 = 4 ; RC = (1/3 + 1/3) = 2/3.
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

        // singletons: each |V|=1, L_in=0 → KKM = 2(6−6) − 0 = 0.
        assert!(kkm_sing.abs() < 1e-9, "singletons KKM={kkm_sing}");
        // all-in-one: zero cut → RC = 0.
        assert!(rc_one.abs() < 1e-9, "one-community RC={rc_one}");

        // The split beats both degenerate extremes on the opposing objective.
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
        // m = 7. Each community: l_c=3, d_c=7.
        // Q = 2 * (3/7 − (7/14)^2) = 2 * (3/7 − 0.25) = 0.3571428...
        let q = modularity(&g, &split);
        assert!((q - 0.357142857142857).abs() < 1e-9, "Q={q}");
    }

    #[test]
    fn modularity_split_beats_one_community() {
        let g = two_triangles();
        let split: Labels = vec![0, 0, 0, 1, 1, 1];
        let one: Labels = vec![0, 0, 0, 0, 0, 0];
        // one community: l_c = 7 (all edges internal), d_c = 14 → Q = 7/7 − 1 = 0.
        let q_one = modularity(&g, &one);
        assert!(q_one.abs() < 1e-9, "one-community Q={q_one}");
        assert!(modularity(&g, &split) > q_one);
    }
}
