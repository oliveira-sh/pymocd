use super::*;

use rand::{Rng, RngExt};
use std::collections::HashMap;

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
) -> Vec<Genome> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = parents[0].len();
    let mut r = rand::rng();
    let mut children: Vec<Genome> = Vec::with_capacity(pop);

    for _ in 0..pop {
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
        children.push(child);
    }
    children
}

// Micro offspring (Alg. 1 line 7): one-way crossover (prob p_c) + neighbor mutation (rate 1/n).
pub fn micro_offspring(
    g: &Graph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    let mut r = rand::rng();
    let mut children: Vec<Labels> = Vec::with_capacity(pop);

    for _ in 0..pop {
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
            let nbrs = &g.adj[i];
            if !nbrs.is_empty() && r.random_bool(p_mut) {
                let t = nbrs[r.random_range(0..nbrs.len())];
                child[i] = child[t];
            }
        }

        children.push(child);
    }
    children
}

// Local search (Alg. 1 line 11): Louvain first-phase modularity ascent, in place.
// Move set deferred to ref [38]; the paper pins only target + objective (Newman Q).
// ΔQ(move i → c) ∝ w(c) − tot[c]·k_i / m2, with i first removed from its community.

// Bounds runtime only; a no-move sweep normally converges earlier.
const LOCAL_SEARCH_SWEEP_CAP: usize = 64;

pub fn local_search(g: &Graph, labels: &mut Labels) {
    let n = g.n;
    let m2 = g.m2;
    if n == 0 || m2 <= 0.0 {
        return;
    }

    let mut tot: HashMap<i32, f64> = HashMap::new();
    for i in 0..n {
        *tot.entry(labels[i]).or_insert(0.0) += g.deg[i];
    }

    let mut improved = true;
    let mut sweeps = 0usize;
    while improved && sweeps < LOCAL_SEARCH_SWEEP_CAP {
        improved = false;
        sweeps += 1;

        for i in 0..n {
            let ki = g.deg[i];
            if ki == 0.0 {
                continue;
            }
            let ci = labels[i];

            let mut w: HashMap<i32, f64> = HashMap::new();
            for &t in &g.adj[i] {
                *w.entry(labels[t]).or_insert(0.0) += 1.0;
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
