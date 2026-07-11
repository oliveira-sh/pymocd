//! Microcanonical Bernoulli-SBM description length (Peixoto-style), the
//! label-free criterion the SCALE frontier selector minimises.
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
