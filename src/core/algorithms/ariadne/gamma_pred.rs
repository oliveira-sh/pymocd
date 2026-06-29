//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos.

use crate::core::graph::CsrGraph;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap;

/// Cheap structural summary from a label-propagation seed partition.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    /// Estimated mixing: fraction of edges crossing community boundaries.
    pub mu_hat: f64,
    /// Characteristic internal density Σe_c / ΣC(s_c,2) of the seed partition.
    pub density: f64,
    /// Mean community size (n / #communities).
    pub comm_size: f64,
    pub n_comm: usize,
}

/// Asynchronous label propagation: each sweep moves every node (random order)
/// to its most frequent neighbour label, ties broken randomly. Converges to a
/// rough community structure cheaply; used only to read off mixing/density.
fn label_prop(g: &CsrGraph, iters: usize, seed: u64) -> Vec<i32> {
    let n = g.n;
    let mut lab: Vec<i32> = (0..n as i32).collect();
    if n == 0 {
        return lab;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n).collect();
    let mut counts: FxHashMap<i32, u32> = FxHashMap::default();
    for _ in 0..iters {
        order.shuffle(&mut rng);
        let mut changed = false;
        for &u in &order {
            let nbrs = g.neighbors(u);
            if nbrs.is_empty() {
                continue;
            }
            counts.clear();
            for &v in nbrs {
                *counts.entry(lab[v as usize]).or_insert(0) += 1;
            }
            // most frequent label; random pick among ties for symmetry-breaking.
            let mut best = lab[u];
            let mut best_c = 0u32;
            let mut ties = 0u32;
            for (&l, &c) in counts.iter() {
                if c > best_c {
                    best_c = c;
                    best = l;
                    ties = 1;
                } else if c == best_c {
                    ties += 1;
                    if rng.random_range(0..ties) == 0 {
                        best = l;
                    }
                }
            }
            if lab[u] != best {
                lab[u] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    lab
}

/// Extract mixing / density / community-size features from a label-prop seed.
pub fn features(g: &CsrGraph, seed: u64) -> Features {
    let p = label_prop(g, 12, seed);
    let m = g.m.max(1) as f64;
    let mut inter = 0usize;
    for &(u, v) in &g.edges {
        if p[u as usize] != p[v as usize] {
            inter += 1;
        }
    }
    let intra = g.m - inter;
    let mut sizes: FxHashMap<i32, f64> = FxHashMap::default();
    for &c in &p {
        *sizes.entry(c).or_insert(0.0) += 1.0;
    }
    let pairs: f64 = sizes.values().map(|&s| s * (s - 1.0) * 0.5).sum();
    Features {
        mu_hat: inter as f64 / m,
        density: if pairs > 0.0 {
            intra as f64 / pairs
        } else {
            0.0
        },
        comm_size: g.n as f64 / sizes.len().max(1) as f64,
        n_comm: sizes.len(),
    }
}

/// Predict five geometric resolutions centered on the graph's CPM density
/// threshold. Deterministic given `seed`. If the label-prop seed is degenerate
/// (collapsed to ≤2 communities / near-zero density — typical at high mixing
/// where there is little structure to read), fall back to the modularity-
/// standard grid centered at γ=1 so the ensemble still brackets a useful range.
pub fn predict_gammas(g: &CsrGraph, seed: u64) -> Vec<f64> {
    gammas_from_features(&features(g, seed))
}

/// `predict_gammas` split from feature extraction, so a caller that already has
/// the `Features` (e.g. the regime-gated selector, which also reads `mu_hat`)
/// can derive the five γ without a second label-prop pass.
pub fn gammas_from_features(f: &Features) -> Vec<f64> {
    let s = std::f64::consts::SQRT_2;
    if f.n_comm <= 2 || f.density < 0.02 {
        return vec![0.5, 1.0 / s, 1.0, s, 2.0];
    }
    let g0 = f.density.clamp(0.02, 8.0);
    vec![g0 / 2.0, g0 / s, g0, g0 * s, g0 * 2.0]
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn predicts_five_ordered_positive_gammas() {
        // two triangles + bridge
        let edges = vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)];
        let g = crate::core::graph::CsrGraph {
            n: 6,
            m: edges.len(),
            xadj: {
                let mut deg = vec![0u32; 6];
                for &(u, v) in &edges {
                    deg[u as usize] += 1;
                    deg[v as usize] += 1;
                }
                let mut x = vec![0u32; 7];
                for u in 0..6 {
                    x[u + 1] = x[u] + deg[u];
                }
                x
            },
            adj: {
                let mut deg = vec![0u32; 6];
                let mut x = vec![0u32; 7];
                for &(u, v) in &edges {
                    deg[u as usize] += 1;
                    deg[v as usize] += 1;
                }
                for u in 0..6 {
                    x[u + 1] = x[u] + deg[u];
                }
                let mut adj = vec![0u32; x[6] as usize];
                let mut cur: Vec<u32> = x[..6].to_vec();
                for &(u, v) in &edges {
                    adj[cur[u as usize] as usize] = v;
                    cur[u as usize] += 1;
                    adj[cur[v as usize] as usize] = u;
                    cur[v as usize] += 1;
                }
                adj
            },
            deg: vec![0u32; 6],
            edges,
            labels: (0..6).collect(),
        };
        let gv = predict_gammas(&g, 1);
        assert_eq!(gv.len(), 5);
        for w in gv.windows(2) {
            assert!(w[1] > w[0], "gammas not strictly increasing: {gv:?}");
        }
        assert!(gv[0] > 0.0);
    }
}
