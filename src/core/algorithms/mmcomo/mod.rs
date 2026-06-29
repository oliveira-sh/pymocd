//! MMCoMO — "A Macro-Micro Population-Based Co-Evolutionary Multi-Objective
//! Algorithm for Community Detection in Complex Networks" (Zhang, Yang, Yang &
//! Zhang, IEEE Computational Intelligence Magazine). Two co-evolving NSGA-II
//! populations over the (KKM, RC) bi-objective (both minimized, Eq. 1):
//!   - macro: medoid binary representation `b ∈ {0,1}^n` (community centers,
//!     Eq. 2) — exploration; decoded to a partition via the diffusion-kernel
//!     similarity matrix `SM` (Eqs. 3–5).
//!   - micro: vector label representation `l ∈ {1..n}^n` (Eq. 6) — exploitation,
//!     refined by modularity local search.
//! Every `gap` generations they interact: Guidance (Alg. 2) injects decoded
//! macro elites into the micro pool; Influence (Alg. 3) updates `SM` from a
//! micro-elite voting matrix (Eq. 7) and encodes micro elites back to medoids
//! (Eq. 8). Final answer = max-modularity member of the merged rank-1 front.
//!
//! Objective + NSGA-II machinery is reused from `helpers`/`nsga2`; only the
//! medoid representation, the diffusion kernel and the two interaction
//! strategies are new.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::{Individual, fast_non_dominated_sort};
use crate::core::metaheuristics::helpers::objectives::kernel_ratiocut::kkm_ratiocut;
use crate::core::metaheuristics::helpers::operators::get_modularity_from_partition;
use crate::core::metaheuristics::nsga2::{calculate_crowding_distance, select_survivors};
use crate::core::utils::normalize_community_ids;

use ndarray::{Array2, Zip};
use rand::seq::SliceRandom;
use rand::{prelude::*, rng};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;

mod defaults;
pub use defaults::*;

const LOCAL_SEARCH_SWEEPS: usize = 8;
const EXPM_TAYLOR_TERMS: usize = 18;

/// Index-space view of the graph: contiguous node indices `[0, n)`, an
/// adjacency in those indices, and a similarity matrix indexed the same way.
struct Ctx {
    n: usize,
    nodes: Vec<NodeId>, // index -> original NodeId (sorted)
    adj: Vec<Vec<usize>>,
    deg: Vec<usize>,
    two_m: usize, // 2 * |E|
}

impl Ctx {
    fn new(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone(); // sorted in finalize()
        let idx: FxHashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let adj: Vec<Vec<usize>> = nodes
            .iter()
            .map(|&id| graph.neighbors(&id).iter().map(|nb| idx[nb]).collect())
            .collect();
        let deg: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let two_m: usize = deg.iter().sum();
        Ctx { n: nodes.len(), nodes, adj, deg, two_m }
    }
}

/// A macro individual: its medoid genome plus the decoded partition / objectives
/// carried in `ind` so it can be sorted alongside the vector population.
#[derive(Clone)]
struct MacroInd {
    bits: Vec<u8>,
    ind: Individual,
}

// ----------------------------------------------------------------- diffusion SM

/// Matrix exponential `exp(M)` via scaling-and-squaring with a Taylor core
/// (standard, LAPACK-free; reuses the already-present `ndarray`).
// ponytail: ndarray matmul scaling-and-squaring, no new linalg dependency.
fn expm(mut m: Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    if n == 0 {
        return m;
    }
    // ∞-norm of M.
    let mut norm = 0.0f64;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += m[[i, j]].abs();
        }
        if s > norm {
            norm = s;
        }
    }
    // Scale so ||M / 2^s|| ≤ 0.5, Taylor, then square back s times.
    let mut scale = 1.0f64;
    let mut squarings = 0u32;
    while norm / scale > 0.5 {
        scale *= 2.0;
        squarings += 1;
    }
    m.mapv_inplace(|x| x / scale);

    let mut term = Array2::<f64>::eye(n);
    let mut e = Array2::<f64>::eye(n);
    for k in 1..=EXPM_TAYLOR_TERMS {
        term = term.dot(&m);
        term.mapv_inplace(|x| x / k as f64);
        e += &term;
    }
    for _ in 0..squarings {
        e = e.dot(&e);
    }
    e
}

/// Diffusion-kernel similarity matrix `SM = exp(beta·H)`, `H = A − D`
/// (Kondor–Lafferty diffusion kernel). Symmetric, entrywise positive.
// ponytail: dense O(n^2) SM, sparse/Chebyshev approx if large-n needed.
fn diffusion_sm(ctx: &Ctx, beta: f64) -> Array2<f64> {
    let n = ctx.n;
    let mut h = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        h[[i, i]] = -(ctx.deg[i] as f64);
        for &j in &ctx.adj[i] {
            h[[i, j]] = 1.0;
        }
    }
    h.mapv_inplace(|x| x * beta);
    expm(h)
}

// --------------------------------------------------------- medoid decode/encode

/// Decode a medoid genome (Eqs. 3–5): central nodes form their own community;
/// each non-central node joins the central node of maximum membership `U`.
/// Since `U_{i,j}` (Eq. 4) shares the row denominator across `j`, `argmax_j U`
/// reduces to `argmax_j SM[i, CN_j]`. Returns a label map keyed by NodeId.
fn decode(ctx: &Ctx, sm: &Array2<f64>, bits: &[u8]) -> Partition {
    let n = ctx.n;
    let mut is_center: Vec<bool> = bits.iter().map(|&b| b != 0).collect();
    let mut cn: Vec<usize> = (0..n).filter(|&i| is_center[i]).collect();
    if cn.is_empty() {
        // Degenerate genome (no centers): seed the highest-degree node.
        let c = (0..n).max_by_key(|&i| ctx.deg[i]).unwrap_or(0);
        is_center[c] = true;
        cn.push(c);
    }

    let mut part = Partition::default();
    part.reserve(n);
    for i in 0..n {
        if is_center[i] {
            part.insert(ctx.nodes[i], ctx.nodes[i]); // community id = central node id
        } else {
            let mut best = cn[0];
            let mut best_v = sm[[i, cn[0]]];
            for &c in &cn[1..] {
                let v = sm[[i, c]];
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            part.insert(ctx.nodes[i], ctx.nodes[best]);
        }
    }
    part
}

/// Encode a vector partition to a medoid genome (Eq. 8): in each community, the
/// central node maximizes the summed similarity to the rest of the community.
fn encode(ctx: &Ctx, sm: &Array2<f64>, part: &Partition) -> Vec<u8> {
    let n = ctx.n;
    let mut groups: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        groups.entry(part[&ctx.nodes[i]]).or_default().push(i);
    }
    let mut bits = vec![0u8; n];
    for members in groups.values() {
        let mut best = members[0];
        let mut best_v = f64::NEG_INFINITY;
        for &v in members {
            let mut s = 0.0;
            for &u in members {
                if u != v {
                    s += sm[[v, u]];
                }
            }
            if s > best_v {
                best_v = s;
                best = v;
            }
        }
        bits[best] = 1;
    }
    bits
}

// -------------------------------------------------------------------- objectives

fn eval(graph: &Graph, ind: &mut Individual) {
    let (kkm, rc) = kkm_ratiocut(graph, &ind.partition);
    ind.objectives = vec![kkm, rc];
}

fn eval_pop(graph: &Graph, pop: &mut [Individual]) {
    pop.par_iter_mut().for_each(|ind| eval(graph, ind));
}

fn build_macro(ctx: &Ctx, sm: &Array2<f64>, graph: &Graph, bits: Vec<u8>) -> MacroInd {
    let part = decode(ctx, sm, &bits);
    let mut ind = Individual::new(part);
    eval(graph, &mut ind);
    MacroInd { bits, ind }
}

// ------------------------------------------------------------------- selection

/// NSGA-II environment selection over macro individuals, keeping their genome.
fn select_macro(pool: &mut Vec<MacroInd>, pop_size: usize) {
    // fast_non_dominated_sort / crowding only read objectives — sort on a cheap
    // objective-only scratch, then carry ranks back onto the genomes.
    let mut scratch: Vec<Individual> = pool
        .iter()
        .map(|m| {
            let mut x = Individual::new(Partition::default());
            x.objectives = m.ind.objectives.clone();
            x
        })
        .collect();
    fast_non_dominated_sort(&mut scratch);
    calculate_crowding_distance(&mut scratch);
    for (m, s) in pool.iter_mut().zip(scratch.iter()) {
        m.ind.rank = s.rank;
        m.ind.crowding_distance = s.crowding_distance;
    }
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        pool[a].ind.rank.cmp(&pool[b].ind.rank).then_with(|| {
            pool[b]
                .ind
                .crowding_distance
                .partial_cmp(&pool[a].ind.crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });
    order.truncate(pop_size);
    *pool = order.iter().map(|&i| pool[i].clone()).collect();
}

#[inline]
fn better(rank_a: usize, crowd_a: f64, rank_b: usize, crowd_b: f64) -> bool {
    rank_a < rank_b || (rank_a == rank_b && crowd_a > crowd_b)
}

fn tournament_micro(pop: &[Individual], rng: &mut impl Rng) -> usize {
    let i = rng.random_range(0..pop.len());
    let j = rng.random_range(0..pop.len());
    if better(pop[j].rank, pop[j].crowding_distance, pop[i].rank, pop[i].crowding_distance) {
        j
    } else {
        i
    }
}

fn tournament_macro(pop: &[MacroInd], rng: &mut impl Rng) -> usize {
    let i = rng.random_range(0..pop.len());
    let j = rng.random_range(0..pop.len());
    if better(
        pop[j].ind.rank,
        pop[j].ind.crowding_distance,
        pop[i].ind.rank,
        pop[i].ind.crowding_distance,
    ) {
        j
    } else {
        i
    }
}

// ------------------------------------------------------------- initialization

/// Micro init (Alg. 1, line 1): each node's label = a random neighbor's id
/// (label-propagation seed, ref [11]).
fn init_micro(ctx: &Ctx, graph: &Graph, pop_size: usize) -> Vec<Individual> {
    (0..pop_size)
        .map(|_| {
            let mut rng = rng();
            let mut part = Partition::default();
            part.reserve(ctx.n);
            for i in 0..ctx.n {
                let lab = if ctx.adj[i].is_empty() {
                    ctx.nodes[i]
                } else {
                    ctx.nodes[ctx.adj[i][rng.random_range(0..ctx.adj[i].len())]]
                };
                part.insert(ctx.nodes[i], lab);
            }
            let mut ind = Individual::new(part);
            eval(graph, &mut ind);
            ind
        })
        .collect()
}

/// Macro init (Alg. 1, line 2; ref [46]): half the population's central-node
/// sets are seeded from high-degree candidate nodes, half are random. The
/// number of centers per individual is drawn in `[1, ⌈√n⌉]` to keep the medoid
/// genomes sparse (the representation favours few centers, cf. Fig. 2).
fn init_macro(ctx: &Ctx, sm: &Array2<f64>, graph: &Graph, pop_size: usize) -> Vec<MacroInd> {
    let n = ctx.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| ctx.deg[b].cmp(&ctx.deg[a]));
    let cmax = ((n as f64).sqrt().ceil() as usize).clamp(1, n);

    let mut rng = rng();
    let mut genomes: Vec<Vec<u8>> = Vec::with_capacity(pop_size);
    for k in 0..pop_size {
        let c = rng.random_range(1..=cmax);
        let mut bits = vec![0u8; n];
        if k < pop_size / 2 {
            // High-degree seeded: sample c distinct from the top candidates.
            let cand = (3 * c).min(n);
            let mut pool: Vec<usize> = by_deg[..cand].to_vec();
            pool.shuffle(&mut rng);
            for &i in pool.iter().take(c) {
                bits[i] = 1;
            }
        } else {
            let mut chosen: FxHashSet<usize> = FxHashSet::default();
            while chosen.len() < c {
                chosen.insert(rng.random_range(0..n));
            }
            for i in chosen {
                bits[i] = 1;
            }
        }
        if bits.iter().all(|&b| b == 0) {
            bits[by_deg[0]] = 1;
        }
        genomes.push(bits);
    }
    genomes
        .into_par_iter()
        .map(|b| build_macro(ctx, sm, graph, b))
        .collect()
}

// --------------------------------------------------------------- offspring

/// Macro offspring (Alg. 1, line 5): uniform crossover + per-bit mutation `p_m`.
fn macro_offspring(pool: &[MacroInd], n: usize, p_m: f64) -> Vec<Vec<u8>> {
    let pop = pool.len();
    (0..pop)
        .map(|_| {
            let mut rng = rng();
            let a = tournament_macro(pool, &mut rng);
            let b = tournament_macro(pool, &mut rng);
            let (pa, pb) = (&pool[a].bits, &pool[b].bits);
            let mut child: Vec<u8> =
                (0..n).map(|i| if rng.random_bool(0.5) { pa[i] } else { pb[i] }).collect();
            for c in child.iter_mut() {
                if rng.random_bool(p_m) {
                    *c ^= 1;
                }
            }
            child
        })
        .collect()
}

/// One-way crossover (ref [33]): copy one whole community of `p2` into `p1`.
fn one_way_crossover(ctx: &Ctx, p1: &Partition, p2: &Partition, rng: &mut impl Rng) -> Partition {
    let mut child = p1.clone();
    let j = rng.random_range(0..ctx.n);
    let src = p2[&ctx.nodes[j]];
    for i in 0..ctx.n {
        if p2[&ctx.nodes[i]] == src {
            child.insert(ctx.nodes[i], src);
        }
    }
    child
}

/// Neighbor-based mutation (Alg. 1, line 7): each node adopts a random
/// neighbor's label with probability `1/n`.
fn neighbor_mutation(ctx: &Ctx, child: &mut Partition, rng: &mut impl Rng) {
    let p = 1.0 / ctx.n as f64;
    for i in 0..ctx.n {
        if !ctx.adj[i].is_empty() && rng.random_bool(p) {
            let t = ctx.adj[i][rng.random_range(0..ctx.adj[i].len())];
            let lab = child[&ctx.nodes[t]];
            child.insert(ctx.nodes[i], lab);
        }
    }
}

/// Micro offspring (Alg. 1, line 7): one-way crossover w.p. `p_c` + neighbor mutation.
fn micro_offspring(pool: &[Individual], ctx: &Ctx, graph: &Graph, p_c: f64) -> Vec<Individual> {
    let parts: Vec<Partition> = (0..pool.len())
        .map(|_| {
            let mut rng = rng();
            let a = tournament_micro(pool, &mut rng);
            let mut child = if rng.random_bool(p_c) {
                let b = tournament_micro(pool, &mut rng);
                one_way_crossover(ctx, &pool[a].partition, &pool[b].partition, &mut rng)
            } else {
                pool[a].partition.clone()
            };
            neighbor_mutation(ctx, &mut child, &mut rng);
            child
        })
        .collect();
    let mut off: Vec<Individual> = parts.into_iter().map(Individual::new).collect();
    eval_pop(graph, &mut off);
    off
}

// ----------------------------------------------------------- local search

/// Modularity-improving local search (ref [38]): Louvain-style first-phase node
/// moves applied to the rank-1 micro individuals. Each node moves to the
/// neighbouring community with the largest modularity gain until no sweep
/// improves anything.
fn modularity_local_search(ctx: &Ctx, comm: &mut [i32]) {
    let m2 = ctx.two_m as f64;
    if m2 == 0.0 {
        return;
    }
    let mut tot: FxHashMap<i32, f64> = FxHashMap::default();
    for i in 0..ctx.n {
        *tot.entry(comm[i]).or_insert(0.0) += ctx.deg[i] as f64;
    }
    let mut improved = true;
    let mut sweeps = 0;
    while improved && sweeps < LOCAL_SEARCH_SWEEPS {
        improved = false;
        sweeps += 1;
        for i in 0..ctx.n {
            if ctx.deg[i] == 0 {
                continue;
            }
            let ci = comm[i];
            let ki = ctx.deg[i] as f64;
            // Edges from i to each neighbouring community.
            let mut w: FxHashMap<i32, f64> = FxHashMap::default();
            for &t in &ctx.adj[i] {
                *w.entry(comm[t]).or_insert(0.0) += 1.0;
            }
            // Remove i from its community.
            *tot.get_mut(&ci).unwrap() -= ki;
            // Louvain gain (relative): w(c) − tot[c]·k_i / 2m; staying is a candidate.
            let mut best_c = ci;
            let mut best_g =
                w.get(&ci).copied().unwrap_or(0.0) - tot.get(&ci).copied().unwrap_or(0.0) * ki / m2;
            for (&c, &wc) in w.iter() {
                if c == ci {
                    continue;
                }
                let g = wc - tot.get(&c).copied().unwrap_or(0.0) * ki / m2;
                if g > best_g + 1e-12 {
                    best_g = g;
                    best_c = c;
                }
            }
            *tot.entry(best_c).or_insert(0.0) += ki;
            if best_c != ci {
                comm[i] = best_c;
                improved = true;
            }
        }
    }
}

fn local_search(ctx: &Ctx, graph: &Graph, micro: &mut [Individual]) {
    micro.par_iter_mut().for_each(|ind| {
        if ind.rank != 1 {
            return;
        }
        let mut comm: Vec<i32> = (0..ctx.n).map(|i| ind.partition[&ctx.nodes[i]]).collect();
        modularity_local_search(ctx, &mut comm);
        let mut part = Partition::default();
        part.reserve(ctx.n);
        for i in 0..ctx.n {
            part.insert(ctx.nodes[i], comm[i]);
        }
        ind.partition = part;
        eval(graph, ind);
    });
}

// --------------------------------------------------------- interaction (Alg 2/3)

/// Guidance (Alg. 2): inject the macro rank-1 elites into the micro population.
/// The elites are identified by the macro pop's non-dominated front (line 1,
/// using its cached objectives, set by the prior `select_macro`), then **freshly
/// decoded with the current SM** (line 5, Eq. 4–5) — the genome's cached
/// birth-time decode can lag `SM` after an Influence update, so re-decode here.
fn guidance(
    ctx: &Ctx,
    sm: &Array2<f64>,
    graph: &Graph,
    macro_pop: &[MacroInd],
    micro: Vec<Individual>,
    micro_off: Vec<Individual>,
    pop_size: usize,
) -> Vec<Individual> {
    let mut pool: Vec<Individual> = macro_pop
        .iter()
        .filter(|m| m.ind.rank == 1)
        .map(|m| {
            let mut ind = Individual::new(decode(ctx, sm, &m.bits));
            eval(graph, &mut ind);
            ind
        })
        .collect();
    pool.extend(micro);
    pool.extend(micro_off);
    select_survivors(&mut pool, pop_size);
    pool
}

/// Influence (Alg. 3): build the micro-elite voting matrix, update `SM` (Eq. 7),
/// encode the micro elites to medoids (Eq. 8) and environment-select them with
/// the macro population and its offspring.
#[allow(clippy::too_many_arguments)]
fn influence(
    ctx: &Ctx,
    sm: &mut Array2<f64>,
    graph: &Graph,
    micro: &[Individual],
    mut macro_pop: Vec<MacroInd>,
    macro_off: Vec<MacroInd>,
    t: usize,
    n_gens: usize,
    pop_size: usize,
) -> Vec<MacroInd> {
    let elites: Vec<&Individual> = micro.iter().filter(|x| x.rank == 1).collect();
    let pf = elites.len().max(1) as f64;

    // Voting matrix SM^v: fraction of elites co-assigning each node pair.
    let n = ctx.n;
    let mut smv = Array2::<f64>::zeros((n, n));
    for e in &elites {
        let mut groups: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
        for i in 0..n {
            groups.entry(e.partition[&ctx.nodes[i]]).or_default().push(i);
        }
        for members in groups.values() {
            for &a in members {
                for &b in members {
                    smv[[a, b]] += 1.0 / pf;
                }
            }
        }
    }

    // Eq. 7: SM* = (1−ρ)·SM + ρ·SM^v, ρ = 0.5·t/gen.
    let rho = 0.5 * t as f64 / n_gens as f64;
    Zip::from(&mut *sm).and(&smv).for_each(|s, &v| *s = (1.0 - rho) * *s + rho * v);

    // Eq. 8: encode each micro elite to a medoid using the updated SM.
    let mut pool: Vec<MacroInd> = elites
        .iter()
        .map(|e| {
            let bits = encode(ctx, sm, &e.partition);
            build_macro(ctx, sm, graph, bits)
        })
        .collect();
    pool.append(&mut macro_pop);
    pool.extend(macro_off);
    select_macro(&mut pool, pop_size);
    pool
}

// ------------------------------------------------------------------- driver

/// Run MMCoMO and return the **max-modularity** member of the merged rank-1
/// front (the paper's ground-truth-free decision rule), normalized so isolated
/// nodes get community `-1`.
#[allow(clippy::too_many_arguments)]
pub fn mmcomo(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Partition {
    let front = run_fronts(graph, pop_size, num_gens, cross_rate, mut_rate, gap, beta);
    let best = front
        .into_iter()
        .max_by(|a, b| {
            get_modularity_from_partition(a, graph)
                .partial_cmp(&get_modularity_from_partition(b, graph))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or_default();
    normalize_community_ids(graph, best)
}

/// Run MMCoMO and return the full merged rank-1 Pareto front (Phase 3,
/// Algorithm 1 line 18) — the candidate set `mmcomo` selects from. Exposed so
/// callers can apply their own decision rule (e.g. best-NMI against a known
/// ground truth, the paper's Table IV rule). Each member is normalized.
#[allow(clippy::too_many_arguments)]
pub fn mmcomo_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<Partition> {
    run_fronts(graph, pop_size, num_gens, cross_rate, mut_rate, gap, beta)
        .into_iter()
        .map(|p| normalize_community_ids(graph, p))
        .collect()
}

/// The co-evolution engine (Algorithm 1). Returns the merged rank-1 front
/// (raw, unnormalized partitions).
#[allow(clippy::too_many_arguments)]
fn run_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<Partition> {
    let ctx = Ctx::new(graph);
    if ctx.n == 0 {
        return vec![Partition::default()];
    }

    let mut sm = diffusion_sm(&ctx, beta);
    let mut micro = init_micro(&ctx, graph, pop_size);
    let mut macro_pop = init_macro(&ctx, &sm, graph, pop_size);
    select_survivors(&mut micro, pop_size); // seed rank/crowding for tournaments
    select_macro(&mut macro_pop, pop_size);

    let gap = gap.max(1);
    for t in 1..=num_gens {
        let micro_off = micro_offspring(&micro, &ctx, graph, cross_rate);
        let macro_off_bits = macro_offspring(&macro_pop, ctx.n, mut_rate);
        let macro_off: Vec<MacroInd> = macro_off_bits
            .into_par_iter()
            .map(|b| build_macro(&ctx, &sm, graph, b))
            .collect();

        if t % gap == 0 {
            micro = guidance(&ctx, &sm, graph, &macro_pop, micro, micro_off, pop_size);
            local_search(&ctx, graph, &mut micro);
            // local search changed objectives → refresh rank/crowding before use.
            fast_non_dominated_sort(&mut micro);
            calculate_crowding_distance(&mut micro);
            macro_pop =
                influence(&ctx, &mut sm, graph, &micro, macro_pop, macro_off, t, num_gens, pop_size);
        } else {
            micro.extend(micro_off);
            select_survivors(&mut micro, pop_size);
            macro_pop.extend(macro_off);
            select_macro(&mut macro_pop, pop_size);
        }
    }

    // Phase 3 — mergence: return the rank-1 front of micro ∪ macro.
    let mut merged: Vec<Individual> = micro;
    merged.extend(macro_pop.into_iter().map(|m| m.ind));
    fast_non_dominated_sort(&mut merged);
    let front: Vec<Partition> =
        merged.into_iter().filter(|x| x.rank == 1).map(|x| x.partition).collect();
    if front.is_empty() {
        vec![Partition::default()]
    } else {
        front
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn sm_is_symmetric_with_positive_diagonal() {
        let g = two_triangles();
        let ctx = Ctx::new(&g);
        let sm = diffusion_sm(&ctx, DEFAULT_BETA);
        for i in 0..ctx.n {
            assert!(sm[[i, i]] > 0.0, "diagonal must be positive");
            for j in 0..ctx.n {
                assert!((sm[[i, j]] - sm[[j, i]]).abs() < 1e-9, "SM must be symmetric");
            }
        }
    }

    #[test]
    fn medoid_decode_recovers_two_communities() {
        let g = two_triangles();
        let ctx = Ctx::new(&g);
        let sm = diffusion_sm(&ctx, DEFAULT_BETA);
        // Centers at node 0 (triangle 1) and node 3 (triangle 2).
        let mut bits = vec![0u8; ctx.n];
        bits[0] = 1;
        bits[3] = 1;
        let p = decode(&ctx, &sm, &bits);
        // Unambiguous members: 1 only touches {0,2}; 4,5 only touch triangle 2.
        assert_eq!(p[&1], p[&0]);
        assert_eq!(p[&4], p[&3]);
        assert_eq!(p[&5], p[&3]);
        assert_ne!(p[&0], p[&3]);
    }

    #[test]
    fn finds_two_community_split() {
        let g = two_triangles();
        let res = mmcomo(&g, 60, 40, DEFAULT_CROSS_RATE, DEFAULT_MUT_RATE, DEFAULT_GAP, DEFAULT_BETA);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
