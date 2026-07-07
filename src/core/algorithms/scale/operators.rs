//! Genetic operators for `scale`.

use crate::core::graph::CsrGraph;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::{Genome, Labels};

const RNG_BASE: u64 = 0x5CA1_E5EED;

/// Deterministic per-(salt, slot) RNG: `scale` guarantees reproducible fronts
/// (fixed internal seed), so every parallel slot derives its stream from the
/// generation salt and its own index instead of thread-local entropy.
pub(super) fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

// Binary tournament: lower Pareto rank wins, ties broken by larger crowding.
#[inline]
fn tournament(ranks: &[usize], crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = ranks.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    if ranks[i] < ranks[j] || (ranks[i] == ranks[j] && crowd[i] >= crowd[j]) {
        i
    } else {
        j
    }
}

// Macro offspring (Alg. 1 line 5): uniform crossover + per-bit mutation.
pub fn macro_offspring(
    parents: &[Genome],
    ranks: &[usize],
    crowd: &[f64],
    p_m: f64,
    salt: u64,
) -> Vec<Genome> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = parents[0].len();
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let b = tournament(ranks, crowd, &mut r);
            let (pa, pb) = (&parents[a], &parents[b]);

            let mut child: Genome = Vec::with_capacity(n);
            for i in 0..n {
                let mut bit = if r.random_bool(0.5) { pa[i] } else { pb[i] };
                if r.random_bool(p_m) {
                    bit ^= 1;
                }
                child.push(bit);
            }

            // an all-zero genome decodes to no communities; force one center bit
            if child.iter().all(|&b| b == 0) && n > 0 {
                let k = r.random_range(0..n);
                child[k] = 1;
            }
            child
        })
        .collect()
}

// Micro offspring (Alg. 1 line 7): one-way crossover (prob p_c) + neighbor mutation (rate 1/n).
pub fn micro_offspring(
    g: &CsrGraph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
    salt: u64,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let mut child: Labels = parents[a].clone();

            // one-way crossover: graft b's community-of-j over a
            if r.random_bool(p_c) && n > 0 {
                let b = tournament(ranks, crowd, &mut r);
                let pb = &parents[b];
                let j = r.random_range(0..n);
                let donor = pb[j];
                for u in 0..n {
                    if pb[u] == donor {
                        child[u] = donor;
                    }
                }
            }

            for i in 0..n {
                let nbrs = g.neighbors(i);
                if !nbrs.is_empty() && r.random_bool(p_mut) {
                    let t = nbrs[r.random_range(0..nbrs.len())] as usize;
                    child[i] = child[t];
                }
            }
            child
        })
        .collect()
}

// Local search (Alg. 1 line 11): Louvain first-phase modularity ascent, in place.
// ΔQ(move i → c) ∝ w(c) − tot[c]·k_i / m2, with i first removed from its community.

// Bounds runtime only; a no-move sweep normally converges earlier.
const LOCAL_SEARCH_SWEEP_CAP: usize = 64;

pub fn local_search(g: &CsrGraph, labels: &mut Labels) {
    let n = g.n;
    let m2 = (2 * g.m) as f64;
    if n == 0 || m2 <= 0.0 {
        return;
    }

    let mut tot: FxHashMap<i32, f64> = FxHashMap::default();
    for i in 0..n {
        *tot.entry(labels[i]).or_insert(0.0) += g.deg[i] as f64;
    }

    let mut improved = true;
    let mut sweeps = 0usize;
    while improved && sweeps < LOCAL_SEARCH_SWEEP_CAP {
        improved = false;
        sweeps += 1;

        for i in 0..n {
            let ki = g.deg[i] as f64;
            if ki == 0.0 {
                continue;
            }
            let ci = labels[i];

            let mut w: FxHashMap<i32, f64> = FxHashMap::default();
            for &t in g.neighbors(i) {
                *w.entry(labels[t as usize]).or_insert(0.0) += 1.0;
            }

            // remove i from its own community before scoring candidates
            if let Some(s) = tot.get_mut(&ci) {
                *s -= ki;
            }

            let mut best_c = ci;
            let mut best_g =
                w.get(&ci).copied().unwrap_or(0.0) - tot.get(&ci).copied().unwrap_or(0.0) * ki / m2;

            for (&c, &wc) in w.iter() {
                if c == ci {
                    continue;
                }
                let g_move = wc - tot.get(&c).copied().unwrap_or(0.0) * ki / m2;
                if g_move > best_g + 1e-12 {
                    best_g = g_move;
                    best_c = c;
                }
            }

            *tot.entry(best_c).or_insert(0.0) += ki;
            if best_c != ci {
                labels[i] = best_c;
                improved = true;
            }
        }
    }
}

// Topology-aware micro offspring (HP-MOCD-style), with the two operators
// independently switchable: `topo_cross` = ensemble (majority-vote) crossover
// across three tournament parents (else the baseline one-way community
// graft); `topo_mut` = neighbour-majority mutation (else the baseline
// random-neighbour label copy).
pub fn micro_offspring_topo(
    g: &CsrGraph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
    salt: u64,
    topo_cross: bool,
    topo_mut: bool,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let mut child: Labels;
            if r.random_bool(p_c) && n > 0 {
                if topo_cross {
                    // ensemble crossover: per-node majority over three parents
                    let b = tournament(ranks, crowd, &mut r);
                    let c = tournament(ranks, crowd, &mut r);
                    let (pa, pb, pc) = (&parents[a], &parents[b], &parents[c]);
                    child = Vec::with_capacity(n);
                    for i in 0..n {
                        let (la, lb, lc) = (pa[i], pb[i], pc[i]);
                        let lab = if lb == lc {
                            lb // majority (covers la==lb==lc too)
                        } else if la == lb || la == lc {
                            la
                        } else {
                            match r.random_range(0..3u8) {
                                0 => la,
                                1 => lb,
                                _ => lc,
                            }
                        };
                        child.push(lab);
                    }
                } else {
                    // baseline one-way crossover: graft b's community-of-j over a
                    child = parents[a].clone();
                    let b = tournament(ranks, crowd, &mut r);
                    let pb = &parents[b];
                    let j = r.random_range(0..n);
                    let donor = pb[j];
                    for u in 0..n {
                        if pb[u] == donor {
                            child[u] = donor;
                        }
                    }
                }
            } else {
                child = parents[a].clone();
            }
            let mut freq: FxHashMap<i32, u32> = FxHashMap::default();
            for i in 0..n {
                let nbrs = g.neighbors(i);
                if !nbrs.is_empty() && r.random_bool(p_mut) {
                    if topo_mut {
                        // neighbour-majority mutation
                        freq.clear();
                        let mut best = child[i];
                        let mut bestc = 0u32;
                        for &v in nbrs {
                            let l = child[v as usize];
                            let e = freq.entry(l).or_insert(0);
                            *e += 1;
                            if *e > bestc {
                                bestc = *e;
                                best = l;
                            }
                        }
                        child[i] = best;
                    } else {
                        // baseline: copy a random neighbour's label
                        let t = nbrs[r.random_range(0..nbrs.len())] as usize;
                        child[i] = child[t];
                    }
                }
            }
            child
        })
        .collect()
}
