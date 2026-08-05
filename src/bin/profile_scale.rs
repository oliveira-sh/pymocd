//! Timing / flamegraph harness for SCALE.
//! Usage: profile_scale [n] [gens] [avg_deg] [mu]
//! Prints wall time and a stable hash of the partition so runs are comparable.

use pymocd::core::algorithms::scale::{
    DEFAULT_BETA, DEFAULT_CROSS_RATE, DEFAULT_GAP, DEFAULT_MACRO_CAP, DEFAULT_MICRO_MUT,
    DEFAULT_MUT_RATE, scale_capped,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

/// Planted partition: `n` nodes in communities of `csize`, `avg_deg * n / 2`
/// unique edges, a `mu` fraction of them crossing communities. Fixed seed.
fn planted(n: usize, avg_deg: usize, mu: f64) -> (Vec<i32>, Vec<(i32, i32)>) {
    const CSIZE: usize = 50;
    let mut r = ChaCha8Rng::seed_from_u64(0xB0BA_CAFE);
    let k = (n / CSIZE).max(1);
    let target = n * avg_deg / 2;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(target * 2);
    let mut edges: Vec<(i32, i32)> = Vec::with_capacity(target);
    while edges.len() < target {
        let u = r.random_range(0..n);
        let cu = u / CSIZE;
        let v = if r.random_bool(mu) {
            let mut c = r.random_range(0..k);
            if c == cu {
                c = (c + 1) % k;
            }
            (c * CSIZE + r.random_range(0..CSIZE)).min(n - 1)
        } else {
            let lo = cu * CSIZE;
            let hi = ((cu + 1) * CSIZE).min(n);
            r.random_range(lo..hi)
        };
        if u == v {
            continue;
        }
        let key = (u.min(v) as u32, u.max(v) as u32);
        if seen.insert(key) {
            edges.push((key.0 as i32, key.1 as i32));
        }
    }
    ((0..n as i32).collect(), edges)
}

/// FNV-1a over the partition, in node order. Deterministic across runs.
fn hash(out: &[(i32, i32)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &(node, comm) in out {
        for b in node.to_le_bytes().iter().chain(comm.to_le_bytes().iter()) {
            h = (h ^ *b as u64).wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| a.get(i).map(String::as_str).unwrap_or(d).to_string();
    let n: usize = arg(1, "10000").parse().unwrap();
    let gens: usize = arg(2, "100").parse().unwrap();
    let avg_deg: usize = arg(3, "10").parse().unwrap();
    let mu: f64 = arg(4, "0.3").parse().unwrap();

    let (nodes, edges) = planted(n, avg_deg, mu);
    let t0 = std::time::Instant::now();
    let out = scale_capped(
        &nodes,
        &edges,
        100,
        gens,
        DEFAULT_CROSS_RATE,
        DEFAULT_MUT_RATE,
        DEFAULT_GAP,
        DEFAULT_BETA,
        DEFAULT_MACRO_CAP,
        DEFAULT_MICRO_MUT,
    );
    let secs = t0.elapsed().as_secs_f64();
    let k = out.iter().map(|&(_, c)| c).collect::<HashSet<_>>().len();
    println!(
        "n={n} m={} gens={gens} threads={} time={secs:.3}s k={k} hash={:016x}",
        edges.len(),
        rayon::current_num_threads(),
        hash(&out)
    );
}
