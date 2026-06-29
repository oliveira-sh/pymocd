//! cpm.rs — Constant Potts Model (CPM) bi-objective for the dense NSGA-II.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CsrGraph;

// Objective: Constant Potts Model (CPM) bi-objective decomposition, dense.
//   intra = 1 - (intra_edges / m)                       (maximise internal edges)
//   inter = gamma * Σ_c C(n_c,2) / C(n,2)               (penalise internal pairs)
// Both minimised; CPM quality = 1 - intra - inter = Σ e_c/m - gamma·Σ C(n_c,2)/C(n,2).
// Unlike modularity (which compares e_c to a degree-based configuration null,
// causing the resolution-limit shattering we saw in the ARI numbers), CPM
// compares internal density to the constant gamma — no resolution limit. With
// gamma=1 the effective raw resolution is the graph's edge density
// (m / C(n,2)): "a community must be denser than the graph average".
//
// `cnt` is caller-owned scratch holding per-community node counts.
#[inline]
pub fn cpm_objectives(g: &CsrGraph, part: &[i32], cnt: &mut [f64], gamma: f64) -> [f64; 2] {
    if g.m == 0 || g.n < 2 {
        return [0.0; 2];
    }
    let m = g.m as f64;

    let mut intra_edges = 0usize;
    for &(u, v) in &g.edges {
        intra_edges += (part[u as usize] == part[v as usize]) as usize;
    }

    for x in cnt.iter_mut() {
        *x = 0.0;
    }
    for u in 0..g.n {
        cnt[part[u] as usize] += 1.0;
    }

    // Σ_c C(n_c, 2) = Σ_c n_c(n_c-1)/2 — within-community node pairs.
    let mut pairs = 0.0;
    for &nc in cnt.iter() {
        pairs += nc * (nc - 1.0) * 0.5;
    }
    let total_pairs = g.n as f64 * (g.n as f64 - 1.0) * 0.5;
    let inter = gamma * pairs / total_pairs;

    let intra = 1.0 - (intra_edges as f64 / m);
    [intra, inter]
}

/// CPM scalar of an objective pair (q = 2 − intra − inter), the quantity whose
/// ensemble-wide plateau the adaptive stop watches.
#[inline(always)]
pub fn cpm_q(objectives: &[f64; 2]) -> f64 {
    2.0 - objectives[0] - objectives[1]
}
