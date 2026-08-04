//! SCALE — sparse macro-micro co-evolutionary multi-objective community
//! detection (Zhang, Yang, Yang & Zhang, IEEE CIM), reformulated for
//! near-linear memory/time: sparse-CSR graph plus a sparse local similarity
//! (`sim`) in place of the dense n×n diffusion kernel.
//!
//! # Deliberate divergence from the published algorithm
//!
//! **The local-search step is intentionally NOT implemented.** Algorithm 1 of
//! the paper runs a Louvain-first-phase modularity ascent (line 11, ref [38]) on
//! the rank-1 micro members at every co-evolution step. This implementation
//! removed it outright — the code is deleted, not feature-flagged off.
//!
//! This is a design decision, NOT an oversight or an unfinished port. Do not
//! "restore" it on the assumption that something was missed. Anyone re-adding it
//! is proposing a change of algorithm and owns re-measuring the result. The
//! `topo_mode` bit that selected its `wadj`-weighted variant (bit 5) is deleted
//! along with every other losing operator; see the `operators` module header.
//!
//! # Shipped configuration
//!
//! The no-keyword-argument call runs a MEASURED configuration, not the paper's:
//! `obj_mode = 160` (micro (intra, inter) / macro (KKM, RC)), `topo_mode = 130`
//! (HP-MOCD 4-distinct-parent ensemble crossover + neighbour-majority micro
//! mutation), `cross_rate = 0.7`, `num_gens = 100`, `mut_rate = 0.5`,
//! `micro_mut = 0.5`. Evidence: beats HP-MOCD on 26/33 LFR cells
//! (p = 0.00021) and 8/10 SNAP networks, best arm at mu = 0.3/0.4/0.5 (ARI
//! 0.9997/0.9891/0.8618), beats both pure-objective arms at mu <= 0.5
//! (p < 0.01); only pure (KKM, RC) wins at mu >= 0.6. See the `defaults` module
//! for the per-constant rationale — these values are not free parameters.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::core::graph::CsrGraph;

mod defaults;
mod nsga2;
mod objectives;
mod operators;
mod refine;
mod sim;

pub use defaults::*;

use nsga2::{Obj, crowding_distance, environment_selection, fast_nondominated_sort};
use objectives::{ObjSet, evaluate};
use operators::{MicroOps, macro_offspring, micro_offspring, micro_offspring_topo};
use sim::{decode, encode, init_weights, update_weights};

pub type Labels = Vec<i32>; // micro vector representation
pub type Genome = Vec<u8>; // macro medoid representation

#[derive(Clone)]
struct Mic {
    labels: Labels,
    obj: Obj,
}

#[derive(Clone)]
struct Mac {
    genome: Genome,
    labels: Labels,
    obj: Obj,
}

/// Search configuration: which objective vector each population optimizes.
///
/// The two sides may carry DIFFERENT objective vectors (see
/// `objectives::split_mode`). Every crossing of the co-evolutionary boundary
/// re-evaluates the migrant under the receiving population's vector, so the two
/// fronts stay internally consistent: `guidance` scores macro elites with
/// `micro`, `influence` scores micro elites with `macro_`.
///
/// Survivor selection is NSGA-II crowding on both sides. An NSGA-III
/// reference-point alternative was measured across the whole objective grid and
/// deleted: every winning run used NSGA-II, and with both surviving sets at
/// M = 2 the reference-point machinery had nothing to buy.
#[derive(Clone, Copy)]
struct Cfg {
    micro: ObjSet,
    macro_: ObjSet,
}

impl Cfg {
    fn new(obj_mode: u16) -> Self {
        let (micro, macro_) = objectives::split_mode(obj_mode);
        Cfg { micro, macro_ }
    }

    fn eval_micro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.micro)
    }

    fn eval_macro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.macro_)
    }
}

fn micro_objs(p: &[Mic]) -> Vec<Obj> {
    p.iter().map(|x| x.obj.clone()).collect()
}
fn macro_objs(p: &[Mac]) -> Vec<Obj> {
    p.iter().map(|x| x.obj.clone()).collect()
}

fn ranks_and_crowd(objs: &[Obj]) -> (Vec<usize>, Vec<f64>) {
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
fn init_micro(g: &CsrGraph, pop: usize, cfg: &Cfg) -> Vec<Mic> {
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
            let obj = cfg.eval_micro(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

/// Default `macro_cap`: the historical hardcoded ⌈√n⌉ ceiling. Chosen so that
/// every pre-existing call reproduces its old front bit-for-bit (`1.0 * x` is
/// exact in IEEE-754, so the multiplier vanishes).
pub const DEFAULT_MACRO_CAP: f64 = 1.0;

/// Sanity bounds on the `macro_cap` multiplier. The lower bound only has to be
/// positive — `macro_cmax` floors the result at 1 anyway — and the upper bound
/// keeps `mult * √n` far from the f64→usize saturation edge; `n` is the real
/// ceiling regardless.
const MACRO_CAP_MIN: f64 = 1e-6;
const MACRO_CAP_MAX: f64 = 1e6;

/// Ceiling on the number of centres a macro genome may express: `⌈cap·√n⌉`,
/// still hard-capped at `n` (a genome cannot have more centres than nodes) and
/// floored at 1 (a genome with zero centres decodes to nothing).
///
/// `cap` is clamped to `[MACRO_CAP_MIN, MACRO_CAP_MAX]` and NaN falls back to
/// the default, so a nonsense value degrades to a usable cap instead of 0.
///
/// WHY THIS IS TUNABLE. Heterogeneous objective placement (micro (KKM,RC) /
/// macro (intra,inter,RC)) beats its mirror only while the cap can still
/// express the true community count: measured cap/k_true 2.9→1.0 gives
/// +0.016/+0.019 ARI (p = 1.8e-6 / 0.014) at n ≤ 2000, and the effect vanishes
/// at cap/k_true 0.64/0.45 (+0.004/−0.003, p = 1.0 / 0.70) for n = 5000/10000,
/// where ⌈√n⌉ is below k_true. Raising `cap` restores expressiveness on large
/// graphs at the cost of a coarser initial search.
fn macro_cmax(n: usize, macro_cap: f64) -> usize {
    let mult = if macro_cap.is_nan() {
        DEFAULT_MACRO_CAP
    } else {
        macro_cap.clamp(MACRO_CAP_MIN, MACRO_CAP_MAX)
    };
    // f64→usize saturates, so an extreme product lands on usize::MAX and the
    // clamp below reduces it to n.
    ((mult * (n as f64).sqrt()).ceil() as usize).clamp(1, n.max(1))
}

/// Macro init (Alg. 1 line 2; ref [46]): half high-degree seeded, half random.
/// Centre count in `[1, ⌈macro_cap·√n⌉]` (see `macro_cmax`), high-degree half
/// sampled from the top-`3c`.
///
/// A 2-hop-exclusion alternative (topo bit 6, `A2`) was measured here and
/// deleted: it placed provably better-spread centres and bought +0.0011 ARI.
fn init_macro(g: &CsrGraph, wadj: &[f64], pop: usize, cfg: &Cfg, macro_cap: f64) -> Vec<Mac> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].cmp(&g.deg[a]));
    let cmax = macro_cmax(n, macro_cap);
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
            let obj = cfg.eval_macro(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
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
    cfg: &Cfg,
) -> Vec<Mic> {
    let ranks = fast_nondominated_sort(&macro_objs(macro_pop));
    let mut pool: Vec<Mic> = macro_pop
        .par_iter()
        .enumerate()
        .filter(|(i, _)| ranks[*i] == 1)
        .map(|(_, m)| {
            let labels = decode(g, wadj, &m.genome);
            let obj = cfg.eval_micro(g, &labels);
            Mic { labels, obj }
        })
        .collect();
    pool.extend(micro);
    pool.extend(micro_off);
    select_micro(pool, pop)
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
    cfg: &Cfg,
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
            let obj = cfg.eval_macro(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
        })
        .collect();
    pool.extend(macro_pop);
    pool.extend(macro_off);
    select_macro(pool, pop)
}

/// Algorithm 1, MINUS its local-search line. Returns the rank-1 front of the
/// merged populations, then adds the union-based refinement
/// (`refine::refine_front`).
///
/// Each generation runs macro/micro variation, then either a co-evolution step
/// (`guidance` → `influence`, every `gap` generations) or an ordinary
/// environment selection on each side.
///
/// The published Algorithm 1 additionally runs a Louvain-first-phase modularity
/// ascent on the rank-1 micro members inside the co-evolution step (line 11).
/// That step is intentionally absent here and its code is deleted — see the
/// module header. There is deliberately NO parameter to switch it back on.
#[allow(clippy::too_many_arguments)]
fn run_fronts(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize, // number of generations the search runs; always run in full
    p_c: f64,
    p_m: f64,
    gap: usize,
    _beta: f64, // inert in the sparse path (the dense diffusion β); kept for API parity
    do_refine: bool, // apply the union-based tiny-community refinement to the front
    topo_mode: u8, // operator bitmask; see the `operators` module bit table
    obj_mode: u16, // objective set (see `objectives::ObjSet::from_u8`)
    macro_cap: f64, // multiplier on the macro genome's ⌈√n⌉ centre ceiling (see `macro_cmax`)
    micro_mut: f64, // per-node micro mutation probability; <= 0 selects the 1/n default
) -> Vec<Labels> {
    if g.n == 0 {
        return vec![Vec::new()];
    }
    let gap = gap.max(1);
    let micro_ops = MicroOps::from_topo(topo_mode);
    let cfg = Cfg::new(obj_mode);
    let mut wadj = init_weights(g);
    let mut micro = init_micro(g, pop, &cfg);
    let mut macro_pop = init_macro(g, &wadj, pop, &cfg, macro_cap);

    for t in 1..=num_gens {
        let (mr, mc) = ranks_and_crowd(&micro_objs(&micro));
        let mlabels: Vec<Labels> = micro.iter().map(|x| x.labels.clone()).collect();
        let micro_off: Vec<Mic> = if micro_ops.any() {
            micro_offspring_topo(
                g,
                &mlabels,
                &mr,
                &mc,
                p_c,
                micro_mut,
                2 * t as u64,
                micro_ops,
            )
        } else {
            micro_offspring(g, &mlabels, &mr, &mc, p_c, micro_mut, 2 * t as u64)
        }
        .into_par_iter()
        .map(|l| {
            let obj = cfg.eval_micro(g, &l);
            Mic { labels: l, obj }
        })
        .collect();

        let (ar, ac) = ranks_and_crowd(&macro_objs(&macro_pop));
        let agen: Vec<Genome> = macro_pop.iter().map(|x| x.genome.clone()).collect();
        let macro_off: Vec<Mac> = macro_offspring(&agen, &ar, &ac, p_m, 2 * t as u64 + 1)
            .into_par_iter()
            .map(|gn| {
                let labels = decode(g, &wadj, &gn);
                let obj = cfg.eval_macro(g, &labels);
                Mac {
                    genome: gn,
                    labels,
                    obj,
                }
            })
            .collect();

        if t % gap == 0 {
            micro = guidance(g, &wadj, &macro_pop, micro, micro_off, pop, &cfg);
            macro_pop = influence(
                g, &mut wadj, &micro, macro_pop, macro_off, t, num_gens, pop, &cfg,
            );
        } else {
            micro.extend(micro_off);
            micro = select_micro(micro, pop);
            macro_pop.extend(macro_off);
            macro_pop = select_macro(macro_pop, pop);
        }
    }

    // Mergence: rank-1 of micro ∪ macro. When the two populations optimize
    // DIFFERENT vectors their objective values are not commensurable (they need
    // not even share a length), so the macro members are re-scored under the
    // micro vector — one extra pass over `pop` individuals, once per run. The
    // merged front is then a front in a single well-defined objective space, and
    // its size stays comparable to the homogeneous arms.
    let het = cfg.micro != cfg.macro_;
    let mut labels: Vec<Labels> = Vec::with_capacity(micro.len() + macro_pop.len());
    let mut objs: Vec<Obj> = Vec::with_capacity(micro.len() + macro_pop.len());
    for m in micro {
        labels.push(m.labels);
        objs.push(m.obj);
    }
    for m in macro_pop {
        let obj = if het {
            cfg.eval_micro(g, &m.labels)
        } else {
            m.obj
        };
        labels.push(m.labels);
        objs.push(obj);
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

    if do_refine {
        refine::refine_front(g, front, cfg.micro)
    } else {
        front
    }
}

/// Map index-space `labels` to `(node_id, community)`: isolated nodes (`deg == 0`)
/// get `-1`, remaining community ids renumbered to `0..k`.
fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
    let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(g.n);
    for (i, &d) in g.deg.iter().enumerate() {
        let comm = if d == 0 {
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
/// label-free normalised-scalarisation selector (`select_best`). Fixed
/// `macro_cap`; see `scale_capped` to tune it.
///
/// Arity-stable shim: every caller that does not care about `macro_cap` keeps
/// this signature, which is also the reference the default-preservation tests
/// compare against. The PyO3 layer goes straight to `scale_capped`, so nothing
/// outside `cfg(test)` calls this today.
#[cfg_attr(not(test), allow(dead_code))]
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
) -> Vec<(i32, i32)> {
    scale_capped(
        nodes,
        edges,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        DEFAULT_MACRO_CAP,
        DEFAULT_MICRO_MUT,
    )
}

/// `scale` with a tunable macro centre ceiling: `macro_cap` multiplies the
/// `⌈√n⌉` bound on how many communities a macro genome may express (see
/// `macro_cmax`). `DEFAULT_MACRO_CAP` reproduces `scale` exactly.
#[allow(clippy::too_many_arguments)]
pub fn scale_capped(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    macro_cap: f64,
    micro_mut: f64,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    // `scale` does not expose the mode knobs, so the shipped configuration's
    // `obj_mode`/`topo_mode`/selection are applied here (see `defaults`).
    let front = run_fronts(
        &g,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        true,
        DEFAULT_TOPO_MODE,
        DEFAULT_OBJ_MODE,
        macro_cap,
        micro_mut,
    );
    let best = select_best(&g, front);
    to_output(&g, &best)
}

/// Label-free selection: MIN-MAX-NORMALISED LINEAR SCALARISATION of all four
/// SCALE objectives over the front.
///
/// 1. Score every member on all FOUR minimised objectives — `(kkm, rc)` from
///    `objectives::kkm_rc` and `(intra, inter)` from `objectives::intra_inter`.
///    All four are computed regardless of which `ObjSet` drove the search: the
///    selector is deliberately independent of the search's objective vector,
///    and that is exactly the form the benchmark validated.
/// 2. Min-max normalise each column ACROSS THE FRONT,
///    `v' = (v − min) / (max − min)`. A column that is constant over the front
///    has zero range and carries no information about which member to pick, so
///    it contributes 0 for every member instead of dividing by zero.
/// 3. Return the member with the smallest sum of the four normalised values.
///
/// THE NORMALISATION IS THE WHOLE POINT and must not be dropped: raw KKM+RC run
/// at magnitude 1e3–1e4 against intra+inter bounded in [0, 1], a 3e3–4e4×
/// imbalance, and the raw variant scores 0.2643 mean gap-to-oracle AMI (30×
/// worse) by collapsing to k = 1 at mu = 0.4.
///
/// Strictly label-free and deterministic; ties are broken by LOWEST INDEX,
/// matching the `np.argmin` of the Python reference implementation.
///
/// Evidence: over a 42-cell selector benchmark on SCALE's own fronts this beats
/// the previously deployed SBM/MDL selector on mean gap-to-oracle AMI (0.0089
/// vs 0.0134, paired Wilcoxon p = 0.020) at 16× lower cost (0.18 s vs 2.91 s),
/// and unlike the SBM/MDL rule it IMPROVES with graph size (n = 10000: 0.0067
/// vs 0.0161).
fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let obj: Vec<[f64; 4]> = front
        .par_iter()
        .map(|p| {
            let (kkm, rc) = objectives::kkm_rc(g, p);
            let (intra, inter) = objectives::intra_inter(g, p);
            [kkm, rc, intra, inter]
        })
        .collect();

    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
    for o in &obj {
        for c in 0..4 {
            if o[c] < lo[c] {
                lo[c] = o[c];
            }
            if o[c] > hi[c] {
                hi[c] = o[c];
            }
        }
    }

    let score = |o: &[f64; 4]| -> f64 {
        let mut s = 0.0;
        for c in 0..4 {
            let rng = hi[c] - lo[c];
            // Guarded: a dead column contributes 0, never NaN.
            if rng > 0.0 {
                s += (o[c] - lo[c]) / rng;
            }
        }
        s
    };

    let mut pick = 0usize;
    let mut best = score(&obj[0]);
    for (j, o) in obj.iter().enumerate().skip(1) {
        let s = score(o);
        // Strict `<`: the FIRST minimum wins, as `np.argmin` does.
        if s < best {
            best = s;
            pick = j;
        }
    }
    front.into_iter().nth(pick).unwrap()
}

/// Full merged (refined) rank-1 front; the candidate set `scale` selects from.
/// Fixed `macro_cap`; see `scale_fronts_capped` to tune it.
///
/// Arity-stable shim: the default-preservation tests use it as the "does not
/// pass the parameter" reference. The PyO3 layer goes straight to
/// `scale_fronts_capped`, so nothing outside `cfg(test)` calls this today.
#[cfg_attr(not(test), allow(dead_code))]
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
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
) -> Vec<Vec<(i32, i32)>> {
    scale_fronts_capped(
        nodes,
        edges,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        refine,
        topo_mode,
        obj_mode,
        DEFAULT_MACRO_CAP,
        DEFAULT_MICRO_MUT,
    )
}

/// `scale_fronts` with a tunable macro centre ceiling: `macro_cap` multiplies
/// the `⌈√n⌉` bound on how many communities a macro genome may express (see
/// `macro_cmax`). `DEFAULT_MACRO_CAP` reproduces `scale_fronts` exactly.
#[allow(clippy::too_many_arguments)]
pub fn scale_fronts_capped(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
) -> Vec<Vec<(i32, i32)>> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(
        &g, pop, num_gens, cross_rate, mut_rate, gap, beta, refine, topo_mode, obj_mode, macro_cap,
        micro_mut,
    )
    .iter()
    .map(|l| to_output(&g, l))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- select_best -----------------------------------------------------
    //
    // The normalisation is the reason the selector works at all: without it the
    // sum is decided by KKM+RC (1e3-1e4) against intra+inter (<= 1), and the raw
    // variant measured 30x worse. This pins BOTH halves of the rule structurally
    // -- normalise across the front, then argmin with a lowest-index tie-break.
    #[test]
    fn selector_normalises_and_breaks_ties_by_lowest_index() {
        let nodes: Vec<i32> = (0..10).collect();
        let g = CsrGraph::from_edges(&nodes, &two_clique_edges());

        // Two IDENTICAL members: every column is constant, so every normalised
        // sum is 0 and the first index must win.
        let split: Labels = (0..g.n).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let one: Labels = vec![0; g.n];
        let sing: Labels = (0..g.n as i32).collect();
        assert_eq!(
            select_best(&g, vec![split.clone(), split.clone()]),
            split,
            "a constant column must contribute 0, not NaN"
        );
        // ...including the degenerate one-member front.
        assert_eq!(select_best(&g, vec![one.clone()]), one);

        // On two K5s bridged once, the planted split beats both degenerate
        // partitions under the normalised sum. The RAW sum does not: KKM alone
        // orders these the other way, so this fails if normalisation is dropped.
        let picked = select_best(&g, vec![one.clone(), sing.clone(), split.clone()]);
        assert_eq!(picked, split, "normalised scalarisation missed the split");
    }

    // Triangle {0,1,2}, triangle {3,4,5}, bridge edge (2,3).
    fn two_triangle_edges() -> Vec<(i32, i32)> {
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    }

    // `k` cliques of size `s` wired into a ring by one bridge per adjacent pair.
    // n = k*s, so the caps genuinely differ (k=12, s=5 -> n=60, cmax 8 vs 31).
    fn ring_of_cliques(k: i32, s: i32) -> (Vec<i32>, Vec<(i32, i32)>) {
        let nodes: Vec<i32> = (0..k * s).collect();
        let mut e = Vec::new();
        for c in 0..k {
            let lo = c * s;
            for a in lo..lo + s {
                for b in (a + 1)..lo + s {
                    e.push((a, b));
                }
            }
            e.push((lo + s - 1, (lo + s) % (k * s)));
        }
        (nodes, e)
    }

    // `w` x `h` 4-neighbour grid: no clean community structure, so the search is
    // genuinely ambiguous and small operator changes move the front. (On a ring
    // of cliques the cliques dominate whatever the operators do, and every bit
    // looks correctly, but uninformatively, inert.)
    fn grid(w: i32, h: i32) -> (Vec<i32>, Vec<(i32, i32)>) {
        let nodes: Vec<i32> = (0..w * h).collect();
        let mut e = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let u = y * w + x;
                if x + 1 < w {
                    e.push((u, u + 1));
                }
                if y + 1 < h {
                    e.push((u, u + w));
                }
            }
        }
        (nodes, e)
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
            true,
            0,
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
                true,
                0,
                0,
            )
        };
        assert_eq!(run(), run());
    }

    // ---- obj_mode --------------------------------------------------------

    // Heterogeneous modes (obj_mode >= 100) give the two populations different
    // objective vectors, so they must survive the mergence step that re-scores
    // macro under the micro vector. 160 is the SHIPPED pair and 106 its mirror.
    #[test]
    fn heterogeneous_objective_modes_are_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [106u16, 160, 166, 100] {
            let run = || {
                scale_fronts(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                )
            };
            let a = run();
            assert!(!a.is_empty(), "obj_mode {obj_mode} produced an empty front");
            assert!(a.iter().all(|f| f.len() == 10));
            assert_eq!(a, run(), "obj_mode {obj_mode} is not deterministic");
        }
    }

    // A homogeneous mode must be reachable both ways: `v` and the heterogeneous
    // encoding of (v, v) are the same configuration, so they must agree exactly.
    // Runs over the whole single-digit range, not just the two live ids, because
    // the deleted ids must still decode to the default on BOTH sides.
    #[test]
    fn heterogeneous_encoding_of_equal_sides_matches_homogeneous() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        let run = |obj_mode: u16| {
            scale_fronts(
                &nodes,
                &edges,
                40,
                15,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                true,
                0,
                obj_mode,
            )
        };
        for v in 0..10u16 {
            assert_eq!(run(v), run(100 + v * 10 + v), "mode {v} != het({v},{v})");
        }
        // The wide range (v >= 1000) is an alias now that no two-digit set is
        // left, so it must agree with the one-digit encoding member for member.
        assert_eq!(run(160), run(1600));
        assert_eq!(run(106), run(1006));
    }

    // The two surviving sets must each drive a real search to a deterministic,
    // non-empty, full-length front, and they must NOT produce the same front --
    // an objective set that does not change the search is not an objective set.
    #[test]
    fn both_objective_sets_search_and_differ() {
        let (nodes, edges) = grid(10, 10);
        let n = nodes.len();
        let run = |obj_mode: u16| {
            scale_fronts(
                &nodes,
                &edges,
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                5,
                DEFAULT_BETA,
                true,
                0,
                obj_mode,
            )
        };
        let kkm_rc = run(0);
        let intra_inter = run(6);
        for (label, a) in [("0", &kkm_rc), ("6", &intra_inter)] {
            assert!(!a.is_empty(), "obj_mode {label}: empty front");
            assert!(a.iter().all(|f| f.len() == n), "obj_mode {label}: partial");
        }
        assert_eq!(kkm_rc, run(0), "obj_mode 0 is not deterministic");
        assert_eq!(intra_inter, run(6), "obj_mode 6 is not deterministic");
        assert_ne!(
            kkm_rc, intra_inter,
            "the two objective sets are the same arm"
        );
        // ...and so is the heterogeneous pairing of them.
        assert_ne!(run(160), kkm_rc);
        assert_ne!(run(160), intra_inter);
    }

    // ---- macro_cap -------------------------------------------------------
    //
    // REGRESSION GUARD. `macro_cap` is only safe to ship because its default
    // reproduces the hardcoded `((n as f64).sqrt().ceil() as usize).clamp(1, n)`
    // EXACTLY -- `1.0 * x` is exact in IEEE-754, so the multiplier must vanish
    // for every n. The old code cannot be called, so this pins the arithmetic
    // structurally against a literal restatement of it.
    #[test]
    fn macro_cmax_default_reproduces_hardcoded_sqrt_ceiling() {
        for n in (1..600usize).chain([1000, 1999, 2000, 5000, 9999, 10_000, 65_536]) {
            let old = ((n as f64).sqrt().ceil() as usize).clamp(1, n);
            assert_eq!(
                macro_cmax(n, DEFAULT_MACRO_CAP),
                old,
                "n={n}: macro_cap=1.0 diverged from the pre-change ceiling"
            );
        }
        assert_eq!(DEFAULT_MACRO_CAP, 1.0);
    }

    // The measured table this parameter exists for: cap 1.0 is ⌈√n⌉, the
    // multiplier scales it, and `n` is still the hard ceiling.
    #[test]
    fn macro_cmax_table() {
        assert_eq!(macro_cmax(250, 1.0), 16);
        assert_eq!(macro_cmax(2000, 1.0), 45);
        assert_eq!(macro_cmax(10_000, 1.0), 100);
        assert_eq!(macro_cmax(10_000, 4.0), 400);
        // Still capped at n, whatever the multiplier claims.
        assert_eq!(macro_cmax(10_000, 1e9), 10_000);
        assert_eq!(macro_cmax(10_000, f64::INFINITY), 10_000);
        assert_eq!(macro_cmax(100, 50.0), 100);
        // Nonsense multipliers must never yield 0 centres.
        for bad in [0.0, -1.0, -1e30, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                macro_cmax(10_000, bad) >= 1,
                "macro_cap={bad} produced a zero ceiling"
            );
            assert!(macro_cmax(1, bad) >= 1);
        }
        // n = 0 must not panic on the clamp (min > max) even though the search
        // early-returns before reaching it.
        assert_eq!(macro_cmax(0, 1.0), 1);
    }

    // End-to-end half of the regression guard: `macro_cap = DEFAULT_MACRO_CAP`
    // must agree with itself AND with the `scale_fronts` wrapper that never
    // mentions the parameter.
    #[test]
    fn default_macro_cap_matches_the_unparameterized_wrapper() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            let legacy = || {
                scale_fronts(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                )
            };
            let capped = |cap: f64| {
                scale_fronts_capped(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                    cap,
                    DEFAULT_MICRO_MUT,
                )
            };
            let a = capped(DEFAULT_MACRO_CAP);
            assert_eq!(
                a,
                capped(DEFAULT_MACRO_CAP),
                "obj_mode {obj_mode}: unstable"
            );
            assert_eq!(
                a,
                legacy(),
                "obj_mode {obj_mode}: default changed the front"
            );
            assert_eq!(legacy(), legacy());
        }
        // Same for the selecting entry point.
        let selected = scale_capped(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            DEFAULT_MACRO_CAP,
            DEFAULT_MICRO_MUT,
        );
        let legacy = scale(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        assert_eq!(
            selected, legacy,
            "scale/scale_capped diverged at the default"
        );
    }

    // Every cap must still drive a real search to a deterministic, non-empty,
    // full-length front -- homogeneous (0) and heterogeneous (160, the shipped
    // arm).
    #[test]
    fn macro_cap_variants_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            for cap in [1.0f64, 2.0, 4.0] {
                let run = || {
                    scale_fronts_capped(
                        &nodes,
                        &edges,
                        40,
                        15,
                        DEFAULT_CROSS_RATE,
                        DEFAULT_MUT_RATE,
                        DEFAULT_GAP,
                        DEFAULT_BETA,
                        true,
                        0,
                        obj_mode,
                        cap,
                        DEFAULT_MICRO_MUT,
                    )
                };
                let a = run();
                assert!(!a.is_empty(), "obj_mode {obj_mode} cap {cap}: empty front");
                assert!(
                    a.iter().all(|f| f.len() == 10),
                    "obj_mode {obj_mode} cap {cap}: partial partition"
                );
                assert_eq!(a, run(), "obj_mode {obj_mode} cap {cap}: nondeterministic");
            }
        }
    }

    // If raising the cap does not change the front, the parameter is not wired
    // through init_macro and the whole experiment it exists for is void. Needs a
    // graph where the ceilings actually differ: n = 60 -> 8 centres vs 31.
    #[test]
    fn macro_cap_changes_the_front() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        assert_eq!(macro_cmax(n, 1.0), 8);
        assert_eq!(macro_cmax(n, 4.0), 31);
        let run = |cap: f64| {
            scale_fronts_capped(
                &nodes,
                &edges,
                40,
                15,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                true,
                0,
                0,
                cap,
                DEFAULT_MICRO_MUT,
            )
        };
        let one = run(1.0);
        let four = run(4.0);
        assert_eq!(one, run(1.0), "cap 1.0 nondeterministic");
        assert_eq!(four, run(4.0), "cap 4.0 nondeterministic");
        assert_ne!(one, four, "macro_cap is not wired through to init_macro");
    }

    // ---- topo_mode operators ---------------------------------------------

    // Both surviving bits, the shipped combination, and the baseline must drive
    // a real search to a deterministic, non-empty, full-length front.
    // `gap = 5` with 20 generations gives four co-evolution steps.
    #[test]
    fn every_topo_bit_is_deterministic_and_nonempty() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        for topo_mode in [0u8, 2, 128, 130] {
            let run = || {
                scale_fronts_capped(
                    &nodes,
                    &edges,
                    40,
                    20,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    5,
                    DEFAULT_BETA,
                    true,
                    topo_mode,
                    0,
                    DEFAULT_MACRO_CAP,
                    DEFAULT_MICRO_MUT,
                )
            };
            let a = run();
            assert!(!a.is_empty(), "topo {topo_mode}: empty");
            assert!(
                a.iter().all(|f| f.len() == n),
                "topo {topo_mode}: partial partition"
            );
            assert_eq!(a, run(), "topo {topo_mode}: nondeterministic");
        }
    }

    // A bit that is wired up but inert looks exactly like "the operator does not
    // help", so every SURVIVING bit must be shown to move the end-to-end front,
    // and every DELETED bit must be shown NOT to -- old benchmark rows recording
    // those masks need them to stay no-ops rather than be quietly rebound.
    #[test]
    fn only_the_two_live_topo_bits_change_the_front() {
        let (nodes, edges) = grid(10, 10);
        let run = |topo_mode: u8| {
            scale_fronts_capped(
                &nodes,
                &edges,
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                5,
                DEFAULT_BETA,
                true,
                topo_mode,
                0,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
            )
        };
        let base = run(0);
        for bit in [2u8, 128, 130] {
            assert_ne!(base, run(bit), "topo bit {bit} is inert end to end");
        }
        for dead in [1u8, 4, 8, 16, 32, 64] {
            assert_eq!(
                base,
                run(dead),
                "deleted topo bit {dead} still does something"
            );
            assert_eq!(
                run(130),
                run(130 | dead),
                "deleted topo bit {dead} changed the shipped mask"
            );
        }
    }

    // The micro bits must NOT leak into the macro/init phases:
    // `MicroOps::from_topo` is what routes micro through the topo body, so a
    // mask with no live micro bit must leave the untouched `micro_offspring` in
    // place.
    #[test]
    fn micro_routing_only_reacts_to_micro_bits() {
        for mask in [0u8, 1, 4, 8, 16, 32, 64] {
            assert!(!MicroOps::from_topo(mask).any(), "mask {mask} routes micro");
            assert_eq!(mask & operators::MICRO_BITS, 0);
        }
        for mask in [2u8, 128, 130] {
            assert!(MicroOps::from_topo(mask).any(), "mask {mask} skips micro");
            assert_eq!(mask & operators::MICRO_BITS, mask);
        }
    }
}
