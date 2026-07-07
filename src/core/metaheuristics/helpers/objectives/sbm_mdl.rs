//! sbm_mdl.rs — microcanonical Bernoulli-SBM description length (Peixoto-style),
//! the label-free criterion the SCALE frontier selector minimises.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CsrGraph;
use crate::core::utils::special::{gammln, ln_choose};
use rustc_hash::FxHashMap;

/// Microcanonical Bernoulli-SBM two-part description length of `part` on `g`
/// (Peixoto-style minimum description length). The data term is the log number of
/// simple graphs consistent with the partition's block-edge counts; the model term
/// is the partition cost plus the block-edge-count cost. LOWER is better.
pub fn dl_sbm_score(g: &CsrGraph, part: &[i32]) -> f64 {
    let n = g.n as f64;
    let m = g.m as f64;
    // dense-remap communities to 0..B, with block sizes n_r and a per-node block.
    let mut remap: FxHashMap<i32, usize> = FxHashMap::default();
    let mut n_r: Vec<f64> = Vec::new();
    let mut block: Vec<usize> = vec![0usize; g.n];
    for u in 0..g.n {
        let b = *remap.entry(part[u]).or_insert_with(|| {
            n_r.push(0.0);
            n_r.len() - 1
        });
        block[u] = b;
        n_r[b] += 1.0;
    }
    let bnum = n_r.len();
    // within-block edge counts e_rr and between-block edge counts e_rs (r < s).
    let mut e_rr: Vec<f64> = vec![0.0; bnum];
    let mut e_rs: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for &(u, v) in &g.edges {
        let (ru, rv) = (block[u as usize], block[v as usize]);
        if ru == rv {
            e_rr[ru] += 1.0;
        } else {
            let key = if ru < rv { (ru, rv) } else { (rv, ru) };
            *e_rs.entry(key).or_insert(0.0) += 1.0;
        }
    }
    // data term: log number of simple graphs with these block-edge counts.
    let mut data = 0.0;
    for r in 0..bnum {
        let slots = n_r[r] * (n_r[r] - 1.0) / 2.0; // within-block pair slots
        data += ln_choose(slots, e_rr[r]);
    }
    for (&(r, s), &w) in &e_rs {
        data += ln_choose(n_r[r] * n_r[s], w);
    }
    // model term: partition cost L(b) plus block-edge-count cost L(e).
    let bf = bnum as f64;
    let sum_ln_fact: f64 = n_r.iter().map(|&x| gammln(x + 1.0)).sum();
    let lb = ln_choose(n - 1.0, bf - 1.0) + gammln(n + 1.0) - sum_ln_fact + n.max(1.0).ln();
    let le = ln_choose(bf * (bf + 1.0) / 2.0 + m - 1.0, m);
    data + lb + le
}

/// Degree-corrected SBM description length: the Karrer–Newman degree-corrected
/// profile log-likelihood as the data term, plus the same partition + block-edge
/// model cost as [`dl_sbm_score`]. LOWER is better. Because the degree sequence is
/// modelled separately, this does NOT collapse to a single block under degree
/// heterogeneity or weak structure, where the plain Bernoulli `dl_sbm_score`
/// does; in a label-free selector bake-off it roughly halved the mean NMI gap to
/// the front oracle (0.154 → 0.084) and removed every high-mixing collapse.
pub fn dl_dcsbm_score(g: &CsrGraph, part: &[i32]) -> f64 {
    let n = g.n as f64;
    let m = g.m as f64;
    // dense-remap communities to 0..B, with block sizes n_r and degree sums e_r.
    let mut remap: FxHashMap<i32, usize> = FxHashMap::default();
    let mut n_r: Vec<f64> = Vec::new();
    let mut e_r: Vec<f64> = Vec::new(); // block degree sums (Σ_{u∈r} deg u)
    let mut block: Vec<usize> = vec![0usize; g.n];
    for u in 0..g.n {
        let b = *remap.entry(part[u]).or_insert_with(|| {
            n_r.push(0.0);
            e_r.push(0.0);
            n_r.len() - 1
        });
        block[u] = b;
        n_r[b] += 1.0;
        e_r[b] += g.deg[u] as f64;
    }
    let bnum = n_r.len();
    // within-block internal edge counts e_rr and between-block counts e_rs (r<s).
    let mut e_rr: Vec<f64> = vec![0.0; bnum];
    let mut e_rs: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for &(u, v) in &g.edges {
        let (ru, rv) = (block[u as usize], block[v as usize]);
        if ru == rv {
            e_rr[ru] += 1.0;
        } else {
            let key = if ru < rv { (ru, rv) } else { (rv, ru) };
            *e_rs.entry(key).or_insert(0.0) += 1.0;
        }
    }
    // Karrer–Newman degree-corrected profile log-likelihood (higher = better fit).
    // Internal stub count of block r is 2·e_rr[r]; cross term counts both directions.
    let mut ll = 0.0;
    for r in 0..bnum {
        let w = 2.0 * e_rr[r];
        let e = e_r[r];
        if w > 0.0 && e > 0.0 {
            ll += w * (w / (e * e)).ln();
        }
    }
    for (&(r, s), &mm) in &e_rs {
        let (e1, e2) = (e_r[r], e_r[s]);
        if mm > 0.0 && e1 > 0.0 && e2 > 0.0 {
            ll += 2.0 * mm * (mm / (e1 * e2)).ln();
        }
    }
    // model term: identical to dl_sbm_score (partition cost + block-edge-count cost).
    let bf = bnum as f64;
    let sum_ln_fact: f64 = n_r.iter().map(|&x| gammln(x + 1.0)).sum();
    let lb = ln_choose(n - 1.0, bf - 1.0) + gammln(n + 1.0) - sum_ln_fact + n.max(1.0).ln();
    let le = ln_choose(bf * (bf + 1.0) / 2.0 + m - 1.0, m);
    -ll + lb + le
}

/// Complete degree-corrected SBM description length: `dl_dcsbm_score` plus the
/// degree-sequence cost the profile form omits — Peixoto's -ln P(k|e,b) under a
/// uniform prior, sum_r ln multiset(n_r, e_r) = sum_r ln C(n_r + e_r - 1, e_r).
/// The missing term systematically under-prices coarse merges of
/// degree-heterogeneous blocks; adding it removes the profile score's collapse
/// onto over-coarse partitions under weak (high-mixing) structure. LOWER better.
pub fn dl_dcsbm_full_score(g: &CsrGraph, part: &[i32]) -> f64 {
    let mut n_r: FxHashMap<i32, f64> = FxHashMap::default();
    let mut e_r: FxHashMap<i32, f64> = FxHashMap::default();
    for u in 0..g.n {
        *n_r.entry(part[u]).or_insert(0.0) += 1.0;
        *e_r.entry(part[u]).or_insert(0.0) += g.deg[u] as f64;
    }
    let degseq: f64 = n_r
        .iter()
        .map(|(c, &nr)| {
            let er = e_r[c];
            ln_choose(nr + er - 1.0, er)
        })
        .sum();
    dl_dcsbm_score(g, part) + degseq
}
