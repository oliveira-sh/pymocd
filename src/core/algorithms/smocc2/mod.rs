//! SMOCC-II — SMOCC's macro-micro co-evolution with the NSGA-II genetic
//! search replaced by multi-objective particle swarm optimization (MOPSO,
//! Gong et al. IEEE TEVC 2014 lineage, adapted to SMOCC's archive style).
//!
//! Everything except the search operators is SMOCC's, imported not copied:
//! CSR graph, decode/encode, objectives and `obj_mode` split, the
//! elite-consensus `wadj` co-evolution, `refine_front` and `select_best`.
//! Each swarm member is a particle (position, velocity, personal best);
//! leaders come from an external nondominated archive pruned by crowding
//! distance. There is no generational replacement — particles persist.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::core::graph::CsrGraph;

use super::smocc::nsga2::{Obj, crowding_distance, fast_nondominated_sort};
use super::smocc::objectives::{self, ObjSet, evaluate};
use super::smocc::operators::slot_rng;
use super::smocc::refine::refine_front;
use super::smocc::sim::{decode, encode, init_weights, update_weights};
use super::smocc::{Genome, Labels, macro_cmax, select_best, to_output};

mod defaults;
mod pso;

pub use defaults::*;

/// Micro particle: a label vector position with a per-node velocity in [0,1]
/// (probability the node is pulled toward the leader) and one personal best.
struct MicP {
    x: Labels,
    v: Vec<f64>,
    obj: Obj,
    pb: Labels,
    pb_obj: Obj,
}

/// Macro particle: a binary centre genome whose cardinality is fixed at init
/// (c ~ U[1, cmax]); the set-based update swaps centres, never grows or
/// shrinks the set.
struct MacP {
    genome: Genome,
    labels: Labels,
    obj: Obj,
    pb: Genome,
    pb_obj: Obj,
}

#[derive(Clone)]
struct MicA {
    labels: Labels,
    obj: Obj,
}

#[derive(Clone)]
struct MacA {
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

/// Inertia weight, linearly decreasing W_MAX -> W_MIN over the generations.
fn inertia(t: usize, num_gens: usize) -> f64 {
    if num_gens <= 1 {
        return W_MIN;
    }
    W_MAX - (W_MAX - W_MIN) * (t - 1) as f64 / (num_gens - 1) as f64
}

// --- archive maintenance -------------------------------------------------
//
// Pool = old archive + new positions; keep the rank-1 subset, drop exact
// duplicates (same objectives AND same position, first-in-pool kept), prune
// to capacity by crowding distance. All ties break deterministically:
// stable sorts keyed by pool order.

/// Surviving pool indices, in stable pool order.
fn archive_select(objs: &[Obj], same: impl Fn(usize, usize) -> bool, cap: usize) -> Vec<usize> {
    let ranks = fast_nondominated_sort(objs);
    let mut keep: Vec<usize> = (0..objs.len()).filter(|&i| ranks[i] == 1).collect();

    // Exact duplicates never dominate each other, so dedup after the rank-1
    // filter loses nothing. Group by equal objectives first: a duplicate
    // position forces duplicate objectives, and obj-equal groups are tiny, so
    // the exact position comparison stays cheap.
    let mut by_obj: Vec<usize> = keep.clone();
    by_obj.sort_by(|&a, &b| {
        objs[a]
            .partial_cmp(&objs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut dup: HashSet<usize> = HashSet::new();
    let mut run_start = 0;
    for i in 1..=by_obj.len() {
        if i == by_obj.len() || objs[by_obj[i]] != objs[by_obj[run_start]] {
            for a in run_start..i {
                if dup.contains(&by_obj[a]) {
                    continue;
                }
                for b in (a + 1)..i {
                    if !dup.contains(&by_obj[b]) && same(by_obj[a], by_obj[b]) {
                        dup.insert(by_obj[b]);
                    }
                }
            }
            run_start = i;
        }
    }
    keep.retain(|i| !dup.contains(i));

    if keep.len() > cap {
        let kobjs: Vec<Obj> = keep.iter().map(|&i| objs[i].clone()).collect();
        let crowd = crowding_distance(&kobjs, &vec![1usize; kobjs.len()]);
        let mut order: Vec<usize> = (0..keep.len()).collect();
        order.sort_by(|&a, &b| {
            crowd[b]
                .partial_cmp(&crowd[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(cap);
        order.sort_unstable();
        keep = order.into_iter().map(|i| keep[i]).collect();
    }

    debug_assert!(
        keep.iter().all(|&a| {
            keep.iter()
                .all(|&b| a == b || !super::smocc::nsga2::dominates(&objs[b], &objs[a]))
        }),
        "archive contains a dominated member"
    );
    keep
}

fn update_micro_archive(mut arch: Vec<MicA>, fresh: Vec<MicA>, cap: usize) -> Vec<MicA> {
    arch.extend(fresh);
    let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
    let keep = archive_select(&objs, |a, b| arch[a].labels == arch[b].labels, cap);
    let keep: HashSet<usize> = keep.into_iter().collect();
    arch.into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, a)| a)
        .collect()
}

fn update_macro_archive(mut arch: Vec<MacA>, fresh: Vec<MacA>, cap: usize) -> Vec<MacA> {
    arch.extend(fresh);
    let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
    let keep = archive_select(&objs, |a, b| arch[a].genome == arch[b].genome, cap);
    let keep: HashSet<usize> = keep.into_iter().collect();
    arch.into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, a)| a)
        .collect()
}

/// Crowding over an archive; every member is rank 1 by construction.
fn arch_crowd(objs: &[Obj]) -> Vec<f64> {
    crowding_distance(objs, &vec![1usize; objs.len()])
}

// --- swarm initialization (identical to SMOCC's population init) ---------

fn init_micro_swarm(g: &CsrGraph, pop: usize, cfg: &Cfg) -> Vec<MicP> {
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(u64::MAX, k);
            let x: Labels = (0..g.n)
                .map(|i| {
                    let nbrs = g.neighbors(i);
                    if nbrs.is_empty() {
                        i as i32
                    } else {
                        nbrs[r.random_range(0..nbrs.len())] as i32
                    }
                })
                .collect();
            let obj = cfg.eval_micro(g, &x);
            MicP {
                pb: x.clone(),
                pb_obj: obj.clone(),
                v: vec![0.0; g.n],
                x,
                obj,
            }
        })
        .collect()
}

fn init_macro_swarm(
    g: &CsrGraph,
    wadj: &[f64],
    pop: usize,
    cfg: &Cfg,
    macro_cap: f64,
) -> Vec<MacP> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].cmp(&g.deg[a]));
    let cmax = macro_cmax(n, macro_cap);
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(u64::MAX - 1, k);
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
            MacP {
                pb: genome.clone(),
                pb_obj: obj.clone(),
                genome,
                labels,
                obj,
            }
        })
        .collect()
}

// --- co-evolution, adapted to archives -----------------------------------

/// Guidance (macro -> micro): decode the macro archive with the micro
/// objectives, inject into the micro archive (nondominated-merge + prune),
/// and reset the pbests of the worst-crowded micro particles to the
/// injected solutions.
fn guidance(
    g: &CsrGraph,
    wadj: &[f64],
    mac_arch: &[MacA],
    mic_arch: Vec<MicA>,
    mic: &mut [MicP],
    pop: usize,
    cfg: &Cfg,
) -> Vec<MicA> {
    let inj: Vec<MicA> = mac_arch
        .par_iter()
        .map(|a| {
            let labels = decode(g, wadj, &a.genome);
            let obj = cfg.eval_micro(g, &labels);
            MicA { labels, obj }
        })
        .collect();

    let objs: Vec<Obj> = mic.iter().map(|p| p.obj.clone()).collect();
    let ranks = fast_nondominated_sort(&objs);
    let crowd = crowding_distance(&objs, &ranks);
    let mut order: Vec<usize> = (0..mic.len()).collect();
    // Worst first: highest rank, then lowest crowding, then index.
    order.sort_by(|&a, &b| {
        ranks[b]
            .cmp(&ranks[a])
            .then(
                crowd[a]
                    .partial_cmp(&crowd[b])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.cmp(&b))
    });
    for (j, &pi) in order.iter().take(inj.len()).enumerate() {
        mic[pi].pb.clone_from(&inj[j].labels);
        mic[pi].pb_obj.clone_from(&inj[j].obj);
    }

    update_micro_archive(mic_arch, inj, pop)
}

/// Influence (micro -> macro): the micro archive is the elite set. It drives
/// the elite-consensus `wadj` update (schedule rho = 0.5 t / n_gens,
/// unchanged), then each elite is encoded/decoded under the NEW weights and
/// injected into the macro archive. As in SMOCC, previously stored members
/// keep their cached labels/objectives — those remain valid partitions; only
/// newly created entries see the new `wadj`.
#[allow(clippy::too_many_arguments)]
fn influence(
    g: &CsrGraph,
    wadj: &mut [f64],
    mic_arch: &[MicA],
    mac_arch: Vec<MacA>,
    t: usize,
    num_gens: usize,
    pop: usize,
    cfg: &Cfg,
) -> Vec<MacA> {
    let elites: Vec<&Labels> = mic_arch.iter().map(|a| &a.labels).collect();
    let rho = 0.5 * t as f64 / num_gens as f64;
    update_weights(g, wadj, &elites, rho);

    let wadj_ro: &[f64] = wadj;
    let inj: Vec<MacA> = elites
        .par_iter()
        .map(|e| {
            let genome = encode(g, wadj_ro, e);
            let labels = decode(g, wadj_ro, &genome);
            let obj = cfg.eval_macro(g, &labels);
            MacA {
                genome,
                labels,
                obj,
            }
        })
        .collect();
    update_macro_archive(mac_arch, inj, pop)
}

// --- main loop ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_fronts(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    do_refine: bool,
    obj_mode: u16,
    macro_cap: f64,
) -> Vec<Labels> {
    if g.n == 0 {
        return vec![Vec::new()];
    }
    let gap = gap.max(1);
    let turb = if turb.is_nan() {
        DEFAULT_TURB
    } else {
        turb.clamp(0.0, 1.0)
    };
    let cfg = Cfg::new(obj_mode);
    let mut wadj = init_weights(g);

    let mut mic = init_micro_swarm(g, pop, &cfg);
    let mut mac = init_macro_swarm(g, &wadj, pop, &cfg, macro_cap);

    let mut mic_arch = update_micro_archive(
        Vec::new(),
        mic.iter()
            .map(|p| MicA {
                labels: p.x.clone(),
                obj: p.obj.clone(),
            })
            .collect(),
        pop,
    );
    let mut mac_arch = update_macro_archive(
        Vec::new(),
        mac.iter()
            .map(|p| MacA {
                genome: p.genome.clone(),
                labels: p.labels.clone(),
                obj: p.obj.clone(),
            })
            .collect(),
        pop,
    );

    for t in 1..=num_gens {
        let w = inertia(t, num_gens);
        let p_t = (turb * w / W_MAX).clamp(0.0, 1.0);

        let mic_objs: Vec<Obj> = mic_arch.iter().map(|a| a.obj.clone()).collect();
        let crowd = arch_crowd(&mic_objs);
        pso::micro_step(g, &mut mic, &mic_arch, &crowd, &cfg, w, p_t, t);
        mic_arch = update_micro_archive(
            mic_arch,
            mic.iter()
                .map(|p| MicA {
                    labels: p.x.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );

        let mac_objs: Vec<Obj> = mac_arch.iter().map(|a| a.obj.clone()).collect();
        let crowd = arch_crowd(&mac_objs);
        pso::macro_step(g, &wadj, &mut mac, &mac_arch, &crowd, &cfg, w, p_t, t);
        mac_arch = update_macro_archive(
            mac_arch,
            mac.iter()
                .map(|p| MacA {
                    genome: p.genome.clone(),
                    labels: p.labels.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );

        if t % gap == 0 {
            mic_arch = guidance(g, &wadj, &mac_arch, mic_arch, &mut mic, pop, &cfg);
            mac_arch = influence(g, &mut wadj, &mic_arch, mac_arch, t, num_gens, pop, &cfg);
        }
    }

    // Final phase: merge both swarms + archives, rank-1 front, refine.
    let het = cfg.micro != cfg.macro_;
    let cap = 2 * (mic.len() + mac.len());
    let mut labels: Vec<Labels> = Vec::with_capacity(cap);
    let mut objs: Vec<Obj> = Vec::with_capacity(cap);
    for p in mic {
        labels.push(p.x);
        objs.push(p.obj);
    }
    for a in mic_arch {
        labels.push(a.labels);
        objs.push(a.obj);
    }
    for p in mac {
        let obj = if het {
            cfg.eval_micro(g, &p.labels)
        } else {
            p.obj
        };
        labels.push(p.labels);
        objs.push(obj);
    }
    for a in mac_arch {
        let obj = if het {
            cfg.eval_micro(g, &a.labels)
        } else {
            a.obj
        };
        labels.push(a.labels);
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
        refine_front(g, &wadj, front, cfg.micro)
    } else {
        front
    }
}

// --- public API -----------------------------------------------------------

#[cfg_attr(not(test), allow(dead_code))]
pub fn smocc2(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    macro_cap: f64,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let front = run_fronts(
        &g,
        pop,
        num_gens,
        gap,
        turb,
        true,
        DEFAULT_OBJ_MODE,
        macro_cap,
    );
    let best = select_best(&g, front);
    to_output(&g, &best)
}

#[allow(clippy::too_many_arguments)]
pub fn smocc2_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    refine: bool,
    obj_mode: u16,
    macro_cap: f64,
) -> Vec<Vec<(i32, i32)>> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(&g, pop, num_gens, gap, turb, refine, obj_mode, macro_cap)
        .iter()
        .map(|l| to_output(&g, l))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

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

    #[test]
    fn finds_two_community_split() {
        let nodes: Vec<i32> = (0..10).collect();
        let out = smocc2(
            &nodes,
            &two_clique_edges(),
            60,
            40,
            DEFAULT_GAP,
            DEFAULT_TURB,
            DEFAULT_MACRO_CAP,
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
        let out = smocc2(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_GAP,
            DEFAULT_TURB,
            DEFAULT_MACRO_CAP,
        );
        let c: FxHashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&6], -1);
    }

    #[test]
    fn fronts_are_nonempty() {
        let nodes: Vec<i32> = (0..6).collect();
        let fronts = smocc2_fronts(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_GAP,
            DEFAULT_TURB,
            true,
            0,
            DEFAULT_MACRO_CAP,
        );
        assert!(!fronts.is_empty());
        assert!(fronts.iter().all(|f| f.len() == 6));
    }

    #[test]
    fn fronts_are_deterministic() {
        let nodes: Vec<i32> = (0..6).collect();
        let run = || {
            smocc2_fronts(
                &nodes,
                &two_triangle_edges(),
                40,
                20,
                DEFAULT_GAP,
                DEFAULT_TURB,
                true,
                0,
                DEFAULT_MACRO_CAP,
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
                smocc2_fronts(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_GAP,
                    DEFAULT_TURB,
                    true,
                    obj_mode,
                    DEFAULT_MACRO_CAP,
                )
            };
            let a = run();
            assert!(!a.is_empty(), "obj_mode {obj_mode} produced an empty front");
            assert!(a.iter().all(|f| f.len() == 10));
            assert_eq!(a, run(), "obj_mode {obj_mode} is not deterministic");
        }
    }

    /// The set-based macro update must never change a particle's centre count:
    /// c is fixed at init, in [1, cmax], for the particle's whole lifetime.
    #[test]
    fn macro_cardinality_invariant_survives_updates() {
        let (nodes, edges) = ring_of_cliques(8, 5);
        let g = CsrGraph::from_edges(&nodes, &edges);
        let wadj = init_weights(&g);
        let cfg = Cfg::new(0);
        let pop = 12usize;
        let cmax = macro_cmax(g.n, DEFAULT_MACRO_CAP);

        let mut parts = init_macro_swarm(&g, &wadj, pop, &cfg, DEFAULT_MACRO_CAP);
        let cards: Vec<usize> = parts
            .iter()
            .map(|p| p.genome.iter().filter(|&&b| b != 0).count())
            .collect();
        for &c in &cards {
            assert!((1..=cmax).contains(&c), "init cardinality {c} out of range");
        }

        let mut arch = update_macro_archive(
            Vec::new(),
            parts
                .iter()
                .map(|p| MacA {
                    genome: p.genome.clone(),
                    labels: p.labels.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );
        let num_gens = 30usize;
        for t in 1..=num_gens {
            let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
            let crowd = arch_crowd(&objs);
            // High turbulence to stress the swap path.
            pso::macro_step(
                &g,
                &wadj,
                &mut parts,
                &arch,
                &crowd,
                &cfg,
                inertia(t, num_gens),
                0.5,
                t,
            );
            arch = update_macro_archive(
                arch,
                parts
                    .iter()
                    .map(|p| MacA {
                        genome: p.genome.clone(),
                        labels: p.labels.clone(),
                        obj: p.obj.clone(),
                    })
                    .collect(),
                pop,
            );
            for (k, (p, &c0)) in parts.iter().zip(&cards).enumerate() {
                let c = p.genome.iter().filter(|&&b| b != 0).count();
                assert_eq!(c, c0, "gen {t}: particle {k} drifted {c0} -> {c}");
                let cp = p.pb.iter().filter(|&&b| b != 0).count();
                assert_eq!(cp, c0, "gen {t}: particle {k} pbest drifted");
            }
        }
    }

    /// The archive is the leader pool: it must never hold a dominated member,
    /// must respect its capacity, and must not hold exact duplicates.
    #[test]
    fn archive_never_contains_a_dominated_member() {
        // Deterministic LCG so the test needs no rng plumbing.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as usize
        };
        for case in 0..100 {
            let cap = 1 + case % 12;
            let mut arch: Vec<MicA> = Vec::new();
            for _round in 0..4 {
                let lattice = 2 + case % 7;
                let fresh: Vec<MicA> = (0..(1 + next() % 20))
                    .map(|_| {
                        let o = vec![(next() % lattice) as f64, (next() % lattice) as f64];
                        let labels: Labels = (0..4).map(|_| (next() % 3) as i32).collect();
                        MicA { labels, obj: o }
                    })
                    .collect();
                arch = update_micro_archive(arch, fresh, cap);
                assert!(arch.len() <= cap, "case {case}: archive over capacity");
                for a in 0..arch.len() {
                    for b in 0..arch.len() {
                        if a != b {
                            assert!(
                                !super::super::smocc::nsga2::dominates(
                                    &arch[b].obj,
                                    &arch[a].obj
                                ),
                                "case {case}: dominated member survived"
                            );
                            assert!(
                                !(arch[a].obj == arch[b].obj && arch[a].labels == arch[b].labels),
                                "case {case}: exact duplicate survived"
                            );
                        }
                    }
                }
            }
        }
    }
}
