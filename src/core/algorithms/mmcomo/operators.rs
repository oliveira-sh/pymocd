use super::*;

use rand::{Rng, RngExt};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Binary tournament selection (NSGA-II rule, paper ref [13]): of two uniformly
// drawn candidates, prefer the one with the lower (better) Pareto rank; break
// ties by the larger crowding distance. This is the parent-selection primitive
// used by both the macro and micro offspring routines.
// ---------------------------------------------------------------------------
#[inline]
fn tournament(ranks: &[usize], crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = ranks.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    // i wins ties (rank equal AND crowd equal) by being tested first.
    if ranks[i] < ranks[j] || (ranks[i] == ranks[j] && crowd[i] >= crowd[j]) {
        i
    } else {
        j
    }
}

// ---------------------------------------------------------------------------
// Macro offspring — Algorithm 1, line 5 (ref [46]):
//   "the offspring of the macro-population ... is generated via uniform
//    crossover and bitwise mutation with probability p_m."
//
// For each of |P_ma| children:
//   1. pick two parents by binary tournament (rank, then crowding),
//   2. UNIFORM crossover: each of the n bits is copied from parent A or parent
//      B with probability 0.5 independently,
//   3. BITWISE mutation: each of the n bits is independently flipped (XOR 1)
//      with probability p_m  (per-bit Bernoulli, NOT one flip per individual).
//
// Degenerate guard: a medoid genome with no central bits decodes to an empty
// community set, so an all-zero child has one random bit forced to 1.
// ---------------------------------------------------------------------------
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
            // uniform crossover
            let mut bit = if r.random_bool(0.5) { pa[i] } else { pb[i] };
            // bitwise mutation (per-bit flip with prob p_m)
            if r.random_bool(p_m) {
                bit ^= 1;
            }
            child.push(bit);
        }

        // guard against an all-zero (center-less) genome
        if child.iter().all(|&b| b == 0) && n > 0 {
            let k = r.random_range(0..n);
            child[k] = 1;
        }
        children.push(child);
    }
    children
}

// ---------------------------------------------------------------------------
// Micro offspring — Algorithm 1, line 7 (ref [33], ref [11]):
//   "generated via one-way crossover [33] with probability p_c and
//    neighbor-based mutation (i.e., the node's label is changed to that of a
//    neighboring node) with probability 1/n."
//
// For each of |P_mi| children:
//   1. pick parent a by binary tournament,
//   2. with probability p_c: ONE-WAY crossover with a second tournament parent
//      b — copy one whole community of b into a's label vector: draw a random
//      node j, take its label L = b[j], and set child[u] = L for every node u
//      with b[u] == L (i.e. graft b's community-of-j over a),
//      otherwise: clone a,
//   3. NEIGHBOR mutation: each node, independently with probability 1/n, adopts
//      the label of a uniformly-chosen neighbour (isolated nodes are skipped).
// ---------------------------------------------------------------------------
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

        // one-way crossover with probability p_c
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

        // neighbor-based mutation at rate 1/n
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

// ---------------------------------------------------------------------------
// Local search — Algorithm 1, line 11 (ref [38]), modularity-improving.
//
// The paper pins only the target (rank-1 micro members) and the objective
// (Newman modularity Q); the move set is deferred to ref [38]. This is the
// defensible Louvain first-phase proxy named in the spec: repeatedly sweep all
// nodes, moving each to the neighbouring community that yields the largest
// positive modularity gain, until a full sweep makes no move (or a small safety
// cap is hit). Operates in place on the label vector.
//
// Standard Louvain single-node gain (unweighted): with k_i = deg(i),
// tot[c] = Σ_{v∈c} deg(v), and w(c) = #edges from i into community c,
//   ΔQ(move i → c) ∝ w(c) − tot[c]·k_i / m2          (m2 = 2|E|)
// Node i is first removed from its own community before the candidates (its own
// community included, so "stay" is the no-gain baseline) are scored.
// ---------------------------------------------------------------------------

// Safety cap on full sweeps. The paper does not pin an iteration count (ref [38]);
// this only bounds runtime — convergence (no-move sweep) normally stops earlier.
const LOCAL_SEARCH_SWEEP_CAP: usize = 64;

pub fn local_search(g: &Graph, labels: &mut Labels) {
    let n = g.n;
    let m2 = g.m2;
    if n == 0 || m2 <= 0.0 {
        return;
    }

    // tot[c] = total degree of community c.
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

            // edges from i to each neighbouring community
            let mut w: HashMap<i32, f64> = HashMap::new();
            for &t in &g.adj[i] {
                *w.entry(labels[t]).or_insert(0.0) += 1.0;
            }

            // remove i from its own community before scoring candidates
            if let Some(s) = tot.get_mut(&ci) {
                *s -= ki;
            }

            // baseline: staying in ci
            let mut best_c = ci;
            let mut best_g = w.get(&ci).copied().unwrap_or(0.0)
                - tot.get(&ci).copied().unwrap_or(0.0) * ki / m2;

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

            // insert i into the chosen community
            *tot.entry(best_c).or_insert(0.0) += ki;
            if best_c != ci {
                labels[i] = best_c;
                improved = true;
            }
        }
    }
}
