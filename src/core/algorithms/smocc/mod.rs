//! SMOCC — Sparse Multi-Objective Co-evolutionary Community detection,
//! a macro-micro co-evolutionary detector (Zhang et al., IEEE CIM lineage)
//! reformulated over a CSR graph and a sparse edge similarity for
//! near-linear memory/time.
//!
//! The paper's Louvain local-search step is deliberately deleted, not
//! feature-flagged: re-adding it is a change of algorithm and must be
//! re-measured. Shipped defaults are the measured winners; see `defaults`.
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
pub(crate) mod nsga2;
pub(crate) mod objectives;
pub(crate) mod operators;
pub(crate) mod refine;
pub(crate) mod sim;

pub use defaults::*;

use nsga2::{Obj, crowding_distance, environment_selection, fast_nondominated_sort};
use objectives::{ObjSet, evaluate};
use operators::{MicroOps, macro_offspring, micro_offspring, micro_offspring_topo};
use sim::{decode, encode, init_weights, update_weights};

pub type Labels = Vec<i32>;
pub type Genome = Vec<u8>;

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

pub const DEFAULT_MACRO_CAP: f64 = 1.0;

const MACRO_CAP_MIN: f64 = 1e-6;
const MACRO_CAP_MAX: f64 = 1e6;

pub(crate) fn macro_cmax(n: usize, macro_cap: f64) -> usize {
    let mult = if macro_cap.is_nan() {
        DEFAULT_MACRO_CAP
    } else {
        macro_cap.clamp(MACRO_CAP_MIN, MACRO_CAP_MAX)
    };
    ((mult * (n as f64).sqrt()).ceil() as usize).clamp(1, n.max(1))
}

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

    let rho = 0.5 * t as f64 / n_gens as f64;
    update_weights(g, wadj, &elites, rho);

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

#[allow(clippy::too_many_arguments)]
fn run_fronts(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize,
    p_c: f64,
    p_m: f64,
    gap: usize,
    _beta: f64,
    do_refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
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
        refine::refine_front(g, &wadj, front, cfg.micro)
    } else {
        front
    }
}

pub(crate) fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
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

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn smocc(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<(i32, i32)> {
    smocc_capped(
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

#[allow(clippy::too_many_arguments)]
pub fn smocc_capped(
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

pub(crate) fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
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
        if s < best {
            best = s;
            pick = j;
        }
    }
    front.into_iter().nth(pick).unwrap()
}

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn smocc_fronts(
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
    smocc_fronts_capped(
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

#[allow(clippy::too_many_arguments)]
pub fn smocc_fronts_capped(
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

    #[test]
    fn selector_normalises_and_breaks_ties_by_lowest_index() {
        let nodes: Vec<i32> = (0..10).collect();
        let g = CsrGraph::from_edges(&nodes, &two_clique_edges());

        let split: Labels = (0..g.n).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let one: Labels = vec![0; g.n];
        let sing: Labels = (0..g.n as i32).collect();
        assert_eq!(
            select_best(&g, vec![split.clone(), split.clone()]),
            split,
            "a constant column must contribute 0, not NaN"
        );
        assert_eq!(select_best(&g, vec![one.clone()]), one);

        let picked = select_best(&g, vec![one.clone(), sing.clone(), split.clone()]);
        assert_eq!(picked, split, "normalised scalarisation missed the split");
    }

    fn two_triangle_edges() -> Vec<(i32, i32)> {
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    }

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
        let out = smocc(
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
        let nodes: Vec<i32> = (0..7).collect();
        let out = smocc(
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
        let fronts = smocc_fronts(
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
            smocc_fronts(
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

    #[test]
    fn heterogeneous_objective_modes_are_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [106u16, 160, 166, 100] {
            let run = || {
                smocc_fronts(
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

    #[test]
    fn heterogeneous_encoding_of_equal_sides_matches_homogeneous() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        let run = |obj_mode: u16| {
            smocc_fronts(
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
        assert_eq!(run(160), run(1600));
        assert_eq!(run(106), run(1006));
    }

    #[test]
    fn both_objective_sets_search_and_differ() {
        let (nodes, edges) = grid(10, 10);
        let n = nodes.len();
        let run = |obj_mode: u16| {
            smocc_fronts(
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
        assert_ne!(run(160), kkm_rc);
        assert_ne!(run(160), intra_inter);
    }

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

    #[test]
    fn macro_cmax_table() {
        assert_eq!(macro_cmax(250, 1.0), 16);
        assert_eq!(macro_cmax(2000, 1.0), 45);
        assert_eq!(macro_cmax(10_000, 1.0), 100);
        assert_eq!(macro_cmax(10_000, 4.0), 400);
        assert_eq!(macro_cmax(10_000, 1e9), 10_000);
        assert_eq!(macro_cmax(10_000, f64::INFINITY), 10_000);
        assert_eq!(macro_cmax(100, 50.0), 100);
        for bad in [0.0, -1.0, -1e30, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                macro_cmax(10_000, bad) >= 1,
                "macro_cap={bad} produced a zero ceiling"
            );
            assert!(macro_cmax(1, bad) >= 1);
        }
        assert_eq!(macro_cmax(0, 1.0), 1);
    }

    #[test]
    fn default_macro_cap_matches_the_unparameterized_wrapper() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            let legacy = || {
                smocc_fronts(
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
                smocc_fronts_capped(
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
        let selected = smocc_capped(
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
        let legacy = smocc(
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
            "smocc/smocc_capped diverged at the default"
        );
    }

    #[test]
    fn macro_cap_variants_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            for cap in [1.0f64, 2.0, 4.0] {
                let run = || {
                    smocc_fronts_capped(
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

    #[test]
    fn macro_cap_changes_the_front() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        assert_eq!(macro_cmax(n, 1.0), 8);
        assert_eq!(macro_cmax(n, 4.0), 31);
        let run = |cap: f64| {
            smocc_fronts_capped(
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

    #[test]
    fn every_topo_bit_is_deterministic_and_nonempty() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        for topo_mode in [0u8, 2, 128, 130] {
            let run = || {
                smocc_fronts_capped(
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

    #[test]
    fn only_the_two_live_topo_bits_change_the_front() {
        let (nodes, edges) = grid(10, 10);
        let run = |topo_mode: u8| {
            smocc_fronts_capped(
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
