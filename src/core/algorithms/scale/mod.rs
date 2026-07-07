//! SCALE — Sparse macro-micro Co-evolutionary multi-objective community detection.
//! A scalable reformulation of the dense macro-micro co-evolutionary
//! community detection (Zhang, Yang, Yang & Zhang, IEEE CIM), re-engineered for
//! near-linear memory/time: sparse-CSR graph, the dense n×n diffusion-kernel
//! similarity replaced by a sparse/approximate local similarity (`sim`). The EA structure (macro medoid
//! population + micro label population co-evolving via guidance/influence over the
//! KKM/RC bi-objective) is unchanged from `mmcomo`; only the O(n²) similarity
//! machinery and the dense graph are replaced.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::seq::SliceRandom;
use rand::RngExt;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::core::graph::CsrGraph;
use crate::core::metaheuristics::helpers::objectives::sbm_mdl::dl_sbm_score;

mod defaults;
mod nsga2;
mod objectives;
mod operators;
mod refine;
mod sim;
mod stats;

pub use defaults::*;

use nsga2::{crowding_distance, environment_selection, fast_nondominated_sort};
use objectives::kkm_rc;
use operators::{local_search, macro_offspring, micro_offspring, micro_offspring_topo};
use stats::welch_p;
use sim::{decode, encode, init_weights, update_weights};

pub type Labels = Vec<i32>; // micro vector representation
pub type Genome = Vec<u8>; // macro medoid representation

#[derive(Clone)]
struct Mic {
    labels: Labels,
    obj: (f64, f64),
}

#[derive(Clone)]
struct Mac {
    genome: Genome,
    labels: Labels,
    obj: (f64, f64),
}

fn micro_objs(p: &[Mic]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}
fn macro_objs(p: &[Mac]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}

fn ranks_and_crowd(objs: &[(f64, f64)]) -> (Vec<usize>, Vec<f64>) {
    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);
    (ranks, crowd)
}

fn select_micro(pool: Vec<Mic>, keep: usize) -> Vec<Mic> {
    let objs = micro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

fn select_macro(pool: Vec<Mac>, keep: usize) -> Vec<Mac> {
    let objs = macro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

/// Micro init (Alg. 1 line 1): each node's label = a random neighbour's id.
fn init_micro(g: &CsrGraph, pop: usize) -> Vec<Mic> {
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = operators::slot_rng(u64::MAX, k);
            let labels: Labels = (0..g.n)
                .map(|i| {
                    let nbrs = g.neighbors(i);
                    if nbrs.is_empty() {
                        i as i32
                    } else {
                        nbrs[r.random_range(0..nbrs.len())] as i32
                    }
                })
                .collect();
            let obj = kkm_rc(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

/// Macro init (Alg. 1 line 2; ref [46]): half high-degree seeded, half random.
/// Centre count in `[1, ⌈√n⌉]`, high-degree half sampled from the top-`3c`.
fn init_macro(g: &CsrGraph, wadj: &[f64], pop: usize) -> Vec<Mac> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].cmp(&g.deg[a]));
    let cmax = ((n as f64).sqrt().ceil() as usize).clamp(1, n);
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = operators::slot_rng(u64::MAX - 1, k);
            let c = r.random_range(1..=cmax);
            let mut genome = vec![0u8; n];
            if k < pop / 2 {
                let cand = (3 * c).min(n);
                let mut poolv: Vec<usize> = by_deg[..cand].to_vec();
                poolv.shuffle(&mut r);
                for &i in poolv.iter().take(c) {
                    genome[i] = 1;
                }
            } else {
                let mut chosen: HashSet<usize> = HashSet::new();
                while chosen.len() < c {
                    chosen.insert(r.random_range(0..n));
                }
                for i in chosen {
                    genome[i] = 1;
                }
            }
            if genome.iter().all(|&b| b == 0) {
                genome[by_deg[0]] = 1;
            }
            let labels = decode(g, wadj, &genome);
            let obj = kkm_rc(g, &labels);
            Mac { genome, labels, obj }
        })
        .collect()
}

/// Guidance (Alg. 2): macro rank-1 elites are freshly decoded with the current
/// edge weights (line 5), then environment-selected with micro + offspring.
fn guidance(
    g: &CsrGraph,
    wadj: &[f64],
    macro_pop: &[Mac],
    micro: Vec<Mic>,
    micro_off: Vec<Mic>,
    pop: usize,
) -> Vec<Mic> {
    let ranks = fast_nondominated_sort(&macro_objs(macro_pop));
    let mut pool: Vec<Mic> = macro_pop
        .par_iter()
        .enumerate()
        .filter(|(i, _)| ranks[*i] == 1)
        .map(|(_, m)| {
            let labels = decode(g, wadj, &m.genome);
            let obj = kkm_rc(g, &labels);
            Mic { labels, obj }
        })
        .collect();
    pool.extend(micro);
    pool.extend(micro_off);
    select_micro(pool, pop)
}

/// Modularity local search (Alg. 1 line 11, ref [38]) on rank-1 micro members.
fn local_search_front(g: &CsrGraph, micro: &mut [Mic]) {
    let ranks = fast_nondominated_sort(&micro_objs(micro));
    micro.par_iter_mut().enumerate().for_each(|(i, m)| {
        if ranks[i] == 1 {
            local_search(g, &mut m.labels);
            m.obj = kkm_rc(g, &m.labels);
        }
    });
}

/// Influence (Alg. 3): micro-elite consensus updates the sparse edge weights
/// (`sim::update_weights`, Eq. 7 analogue), then each elite is re-encoded to a
/// medoid genome and decoded, and environment-selected with macro + offspring.
#[allow(clippy::too_many_arguments)]
fn influence(
    g: &CsrGraph,
    wadj: &mut [f64],
    micro: &[Mic],
    macro_pop: Vec<Mac>,
    macro_off: Vec<Mac>,
    t: usize,
    n_gens: usize,
    pop: usize,
) -> Vec<Mac> {
    let ranks = fast_nondominated_sort(&micro_objs(micro));
    let elites: Vec<&Labels> = micro
        .iter()
        .enumerate()
        .filter(|(i, _)| ranks[*i] == 1)
        .map(|(_, m)| &m.labels)
        .collect();

    // Eq. 7: SM* = (1-rho)*SM + rho*SM^v, rho = 0.5*t/gen — sparse, edge-restricted.
    let rho = 0.5 * t as f64 / n_gens as f64;
    update_weights(g, wadj, &elites, rho);

    // wadj is now fixed for this step; each elite's encode→decode is independent.
    let wadj_ro: &[f64] = wadj;
    let mut pool: Vec<Mac> = elites
        .par_iter()
        .map(|e| {
            let genome = encode(g, wadj_ro, e);
            let labels = decode(g, wadj_ro, &genome);
            let obj = kkm_rc(g, &labels);
            Mac { genome, labels, obj }
        })
        .collect();
    pool.extend(macro_pop);
    pool.extend(macro_off);
    select_macro(pool, pop)
}

/// Algorithm 1. Returns the rank-1 front of the merged populations, then adds the
/// union-based refinement (`refine::refine_front`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn run_fronts(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize,
    p_c: f64,
    p_m: f64,
    gap: usize,
    _beta: f64, // inert in the sparse path (the dense diffusion β); kept for API parity
    adaptive: bool, // adaptive (Welch-t-test plateau) stop; `num_gens` is then the ceiling
    conv_pval: f64, // stop once the window-to-window gain is no longer significant at this level
    do_refine: bool, // apply the union-based tiny-community refinement to the front
    topo_mode: u8,   // bit0 = ensemble crossover, bit1 = neighbour-majority mutation
) -> Vec<Labels> {
    if g.n == 0 {
        return vec![Vec::new()];
    }
    let gap = gap.max(1);
    let mut wadj = init_weights(g);
    let mut micro = init_micro(g, pop);
    let mut macro_pop = init_macro(g, &wadj, pop);
    // Convergence-metric history: the population mean of (KKM + RC), which is
    // minimised and plateaus as the search converges. Used by the adaptive stop.
    let mut history: Vec<f64> = Vec::new();

    for t in 1..=num_gens {
        let (mr, mc) = ranks_and_crowd(&micro_objs(&micro));
        let mlabels: Vec<Labels> = micro.iter().map(|x| x.labels.clone()).collect();
        let micro_off: Vec<Mic> = if topo_mode != 0 {
            micro_offspring_topo(g, &mlabels, &mr, &mc, p_c, 2 * t as u64,
                                 topo_mode & 1 != 0, topo_mode & 2 != 0)
        } else {
            micro_offspring(g, &mlabels, &mr, &mc, p_c, 2 * t as u64)
        }
            .into_par_iter()
            .map(|l| {
                let obj = kkm_rc(g, &l);
                Mic { labels: l, obj }
            })
            .collect();

        let (ar, ac) = ranks_and_crowd(&macro_objs(&macro_pop));
        let agen: Vec<Genome> = macro_pop.iter().map(|x| x.genome.clone()).collect();
        let macro_off: Vec<Mac> = macro_offspring(&agen, &ar, &ac, p_m, 2 * t as u64 + 1)
            .into_par_iter()
            .map(|gn| {
                let labels = decode(g, &wadj, &gn);
                let obj = kkm_rc(g, &labels);
                Mac { genome: gn, labels, obj }
            })
            .collect();

        if t % gap == 0 {
            micro = guidance(g, &wadj, &macro_pop, micro, micro_off, pop);
            local_search_front(g, &mut micro);
            macro_pop = influence(g, &mut wadj, &micro, macro_pop, macro_off, t, num_gens, pop);
        } else {
            micro.extend(micro_off);
            micro = select_micro(micro, pop);
            macro_pop.extend(macro_off);
            macro_pop = select_macro(macro_pop, pop);
        }

        // Adaptive plateau stop: after a warm-up, stop once a Welch t-test no
        // longer finds the last window of the convergence metric significantly
        // better than the previous window.
        if adaptive {
            let sum: f64 = micro.iter().map(|m| m.obj.0 + m.obj.1).sum::<f64>()
                + macro_pop.iter().map(|m| m.obj.0 + m.obj.1).sum::<f64>();
            let cnt = (micro.len() + macro_pop.len()).max(1) as f64;
            history.push(sum / cnt);
            let w = CHECK_EVERY;
            if t >= MIN_GENS && history.len() >= 2 * w && history.len().is_multiple_of(w) {
                let h = history.len();
                if welch_p(&history[h - 2 * w..h - w], &history[h - w..h]) >= conv_pval {
                    break;
                }
            }
        }
    }

    // Mergence: rank-1 of micro ∪ macro.
    let mut labels: Vec<Labels> = Vec::with_capacity(micro.len() + macro_pop.len());
    let mut objs: Vec<(f64, f64)> = Vec::with_capacity(micro.len() + macro_pop.len());
    for m in micro {
        labels.push(m.labels);
        objs.push(m.obj);
    }
    for m in macro_pop {
        labels.push(m.labels);
        objs.push(m.obj);
    }
    let ranks = fast_nondominated_sort(&objs);
    let front: Vec<Labels> = labels
        .into_iter()
        .zip(ranks)
        .filter(|(_, r)| *r == 1)
        .map(|(l, _)| l)
        .collect();
    let front = if front.is_empty() {
        vec![(0..g.n as i32).collect()]
    } else {
        front
    };

    // Union-based refinement: add a tiny-community-merge copy of
    // every member, then non-dominated-sort the union back to rank-1. Refinement
    // only adds candidates, so the returned front is at least as good.
    if do_refine { refine::refine_front(g, front) } else { front }
}

/// Map index-space `labels` to `(node_id, community)`: isolated nodes (`deg == 0`)
/// get `-1`, remaining community ids renumbered to `0..k`.
fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
    let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(g.n);
    for i in 0..g.n {
        let comm = if g.deg[i] == 0 {
            -1
        } else {
            *remap.entry(labels[i]).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        };
        out.push((g.labels[i], comm));
    }
    out
}

/// Single selected partition from the merged (refined) rank-1 front, via the
/// winning label-free selector (`select_best` = min SBM/MDL).
#[allow(clippy::too_many_arguments)]
pub fn scale(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    adaptive: bool,
    conv_pval: f64,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let front = run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta, adaptive, conv_pval, true, 0);
    let best = select_best(&g, front);
    to_output(&g, &best)
}

/// Asymptotic surprise (Traag): m * KL(q || p), q = intra-edge fraction,
/// p = intra-pair fraction. MAXIMISE.
fn asymptotic_surprise(g: &CsrGraph, labels: &Labels) -> f64 {
    let n = g.n as f64;
    let m = g.m as f64;
    if m == 0.0 || n < 2.0 {
        return 0.0;
    }
    let mut nc: FxHashMap<i32, f64> = FxHashMap::default();
    for u in 0..g.n {
        *nc.entry(labels[u]).or_insert(0.0) += 1.0;
    }
    let mut intra = 0.0f64;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            intra += 1.0;
        }
    }
    let t = n * (n - 1.0) / 2.0;
    let n_in: f64 = nc.values().map(|&c| c * (c - 1.0) / 2.0).sum();
    if n_in <= 0.0 || n_in >= t {
        return 0.0;
    }
    let (q, p) = (intra / m, n_in / t);
    let mut kl = 0.0;
    if q > 0.0 {
        kl += q * (q / p).ln();
    }
    if q < 1.0 {
        kl += (1.0 - q) * ((1.0 - q) / (1.0 - p)).ln();
    }
    m * kl
}

fn n_blocks(labels: &Labels) -> usize {
    let mut seen: FxHashMap<i32, ()> = FxHashMap::default();
    for &l in labels {
        seen.entry(l).or_insert(());
    }
    seen.len()
}

/// Label-free selection: SBM-MDL with a Bayesian credible-set Occam tie-break
/// and a surprise fallback on MDL abstention.
///
/// 1. Score every member with the Bernoulli SBM description length; take the
///    argmin.
/// 2. If that optimum does not describe the graph better than the one-block
///    null, the MDL significance test found no structure — return the member
///    of maximum asymptotic surprise instead (a liberal significance
///    criterion that stays near the front oracle in the weak-structure
///    regime where MDL becomes over-conservative).
/// 3. Otherwise, among the members whose description length is within
///    ln 20 nats of the optimum (the Kass–Raftery "strong evidence" bound:
///    models the data cannot strongly distinguish), excluding the null,
///    return the one with the fewest communities (ties: lower DL) — Occam's
///    razor over the credible set.
///
/// All three ingredients are literature constants; nothing is fitted to any
/// benchmark. Strictly label-free; deterministic.
///
/// Edge-holdout predictive likelihood was evaluated as the twin tie-break
/// (min-B credible members): the twins' held-out LL difference is ~1 nat
/// with sign flipping across mask seeds — no signal in expectation, so the
/// deterministic DL tie stays.
fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let dls: Vec<f64> = front.par_iter().map(|p| dl_sbm_score(g, p)).collect();
    let mut i = 0;
    for j in 1..front.len() {
        if dls[j] < dls[i] {
            i = j;
        }
    }
    let trivial = dl_sbm_score(g, &vec![0i32; g.n]);
    if !(dls[i] < trivial - 1e-9) {
        // MDL abstains: no significant structure — argmax surprise.
        let sur: Vec<f64> = front.par_iter().map(|p| asymptotic_surprise(g, p)).collect();
        let mut b = 0;
        for j in 1..front.len() {
            if sur[j] > sur[b] {
                b = j;
            }
        }
        return front.into_iter().nth(b).unwrap();
    }
    let band = dls[i] + 20.0f64.ln();
    let bs: Vec<usize> = front.par_iter().map(n_blocks).collect();
    let mut best: Option<usize> = None;
    for j in 0..front.len() {
        if dls[j] <= band && bs[j] > 1 {
            best = match best {
                None => Some(j),
                Some(k) => {
                    if (bs[j], dls[j]) < (bs[k], dls[k]) {
                        Some(j)
                    } else {
                        Some(k)
                    }
                }
            };
        }
    }
    let pick = best.unwrap_or(i);
    front.into_iter().nth(pick).unwrap()
}

/// Full merged (refined) rank-1 front; the candidate set `scale` selects from.
#[allow(clippy::too_many_arguments)]
pub fn scale_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    adaptive: bool,
    conv_pval: f64,
    refine: bool,
    topo_mode: u8,
) -> Vec<Vec<(i32, i32)>> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta, adaptive, conv_pval, refine, topo_mode)
        .iter()
        .map(|l| to_output(&g, l))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, bridge edge (2,3).
    fn two_triangle_edges() -> Vec<(i32, i32)> {
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    }

    // Two K5 cliques {0..4},{5..9} joined by a single bridge (4,5). Unambiguous
    // 2-block structure that the SBM/MDL selector clearly prefers (the tiny
    // two-triangle graph is MDL-borderline, so it is not used for split tests).
    fn two_clique_edges() -> Vec<(i32, i32)> {
        let mut e = Vec::new();
        for (lo, hi) in [(0, 5), (5, 10)] {
            for a in lo..hi {
                for b in (a + 1)..hi {
                    e.push((a, b));
                }
            }
        }
        e.push((4, 5));
        e
    }

    #[test]
    fn finds_two_community_split() {
        let nodes: Vec<i32> = (0..10).collect();
        let out = scale(
            &nodes,
            &two_clique_edges(),
            60,
            40,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            false,
            CONV_PVAL,
        );
        let c: FxHashMap<i32, i32> = out.into_iter().collect();
        for i in 1..5 {
            assert_eq!(c[&0], c[&i], "clique A node {i} split off");
        }
        for i in 6..10 {
            assert_eq!(c[&5], c[&i], "clique B node {i} split off");
        }
        assert_ne!(c[&0], c[&5], "cliques merged");
    }

    #[test]
    fn isolated_node_gets_minus_one() {
        let nodes: Vec<i32> = (0..7).collect(); // node 6 isolated
        let out = scale(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            false,
            CONV_PVAL,
        );
        let c: FxHashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&6], -1);
    }

    #[test]
    fn fronts_are_nonempty() {
        let nodes: Vec<i32> = (0..6).collect();
        let fronts = scale_fronts(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            false,
            CONV_PVAL,
            true,
            0,
        );
        assert!(!fronts.is_empty());
        assert!(fronts.iter().all(|f| f.len() == 6));
    }

    #[test]
    fn fronts_are_deterministic() {
        let nodes: Vec<i32> = (0..6).collect();
        let run = || {
            scale_fronts(
                &nodes,
                &two_triangle_edges(),
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                false,
                CONV_PVAL,
                true,
                0,
            )
        };
        assert_eq!(run(), run());
    }
}
