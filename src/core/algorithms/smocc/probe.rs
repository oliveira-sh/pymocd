//! Instrumented SMOCC: the same search with per-stage counters, component
//! switches, and alternative similarity media. It produces the ablation and
//! diagnostic evidence reported in the paper; the shipped path in
//! `macro_micro::engine` stays untouched, and a test pins the two together.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;
use std::time::Instant;

use crate::core::algorithms::mmcomo::linalg::diffusion_kernel_csr;
use crate::core::graph::CsrGraph;

use super::config::{Cfg, MicroOps};
use super::front::refine_front_mode;
use super::macro_micro::{init_macro_genomes, init_micro_labels, macro_cmax};
use super::nsga2::{Obj, crowding_distance, environment_selection, fast_nondominated_sort};
use super::objectives::{ObjSet, evaluate};
use super::operators::{macro_offspring_mode, micro_offspring, micro_offspring_topo};
use super::similarity::{centre_count, decode_counted, encode, init_weights, update_weights_floor};
use super::{Genome, Labels};

/// Drop the macro population entirely. The consensus update of the similarity
/// still runs at every transfer generation, since the micro elites alone drive
/// it; removing the update as well is `ABL_NO_W_UPDATE`.
pub const ABL_NO_MACRO: u32 = 1;
pub const ABL_NO_GUIDANCE: u32 = 1 << 1;
pub const ABL_NO_INFLUENCE: u32 = 1 << 2;
pub const ABL_NO_W_UPDATE: u32 = 1 << 3;
/// Drop the agglomerative coarsening from the refinement.
pub const ABL_NO_COARSEN: u32 = 1 << 4;

/// Which object supplies the similarity that macro decoding and encoding read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SimMode {
    /// The shipped per-edge vector `W`, re-fitted to elite consensus.
    Learned,
    /// The dense diffusion kernel restricted to the edges of the graph, held
    /// fixed. Isolates the restriction to `E` from the consensus learning.
    KernelOnEdges,
    /// The full dense kernel with nearest-centre decoding, summed-similarity
    /// encoding and an all-pairs consensus update. `O(n^2)` memory.
    Dense,
}

impl SimMode {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::KernelOnEdges,
            2 => Self::Dense,
            _ => Self::Learned,
        }
    }
}

/// Everything the probe records over one run.
#[derive(Default)]
pub struct Diag {
    pub sweeps: Vec<u32>,
    pub fallback_rounds: Vec<u32>,
    pub centres_init: Vec<u32>,
    pub centres_off: Vec<u32>,
    pub centres_pop: Vec<u32>,
    pub centres_influence: Vec<u32>,
    pub guidance_injected: Vec<u32>,
    pub guidance_survived: Vec<u32>,
    pub influence_injected: Vec<u32>,
    pub influence_survived: Vec<u32>,
    pub front_from_micro: u32,
    pub front_from_macro: u32,
    pub front_from_guidance: u32,
    pub front_size: u32,
    pub front_size_refined: u32,
    pub front4_size: u32,
    pub front4_only: u32,
    pub decode_calls: u64,
    pub t_total: f64,
    pub t_micro: f64,
    pub t_macro: f64,
    pub t_exchange: f64,
    pub t_post: f64,
    pub cmax: u32,
    pub seeds_used: u32,
    /// The final edge similarity, filled only when the graph is small enough
    /// that carrying `2m` doubles back to Python is cheap.
    pub w_final: Vec<f64>,
}

/// The similarity medium. `edge` always holds the edge-restricted view, which
/// the micro operators and the refinement read; `dense` is present only in
/// `SimMode::Dense`, where macro decoding and encoding read it instead.
struct Sim {
    edge: Vec<f64>,
    dense: Option<Vec<f64>>,
    floor: f64,
}

pub fn restrict_pub(g: &CsrGraph, sm: &[f64]) -> Vec<f64> {
    restrict(g, sm)
}

/// `dense_decode` for callers outside this module.
pub fn decode_dense_pub(g: &CsrGraph, sm: &[f64], genome: &Genome) -> Labels {
    dense_decode(g, sm, genome)
}

fn restrict(g: &CsrGraph, sm: &[f64]) -> Vec<f64> {
    let n = g.n;
    (0..n)
        .flat_map(|u| {
            let (s, e) = (g.xadj[u] as usize, g.xadj[u + 1] as usize);
            g.adj[s..e]
                .iter()
                .map(|&v| sm[u * n + v as usize])
                .collect::<Vec<f64>>()
        })
        .collect()
}

fn dense_decode(g: &CsrGraph, sm: &[f64], genome: &Genome) -> Labels {
    let n = g.n;
    let mut cn: Vec<usize> = (0..n).filter(|&i| genome[i] != 0).collect();
    if cn.is_empty() {
        let mut best = 0usize;
        for i in 1..n {
            if g.deg[i] > g.deg[best] {
                best = i;
            }
        }
        cn.push(best);
    }
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut best = cn[0];
            let mut best_v = sm[i * n + cn[0]];
            for &c in &cn[1..] {
                let v = sm[i * n + c];
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            best as i32
        })
        .collect()
}

fn dense_encode(sm: &[f64], labels: &Labels, n: usize) -> Genome {
    let mut genome: Genome = vec![0u8; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pos: rustc_hash::FxHashMap<i32, usize> = rustc_hash::FxHashMap::default();
    for (i, &lab) in labels.iter().enumerate() {
        match pos.get(&lab) {
            Some(&p) => groups[p].push(i),
            None => {
                pos.insert(lab, groups.len());
                groups.push(vec![i]);
            }
        }
    }
    for members in &groups {
        if members.len() == 1 {
            genome[members[0]] = 1;
            continue;
        }
        let mut best = members[0];
        let mut best_sum = f64::NEG_INFINITY;
        for &v in members {
            let mut s = 0.0;
            for &u in members {
                if u != v {
                    s += sm[v * n + u];
                }
            }
            if s > best_sum {
                best_sum = s;
                best = v;
            }
        }
        genome[best] = 1;
    }
    genome
}

fn dense_update(n: usize, sm: &mut [f64], elites: &[&Labels], rho: f64) {
    let pf = elites.len() as f64;
    if pf == 0.0 {
        return;
    }
    sm.par_chunks_mut(n).enumerate().for_each(|(u, row)| {
        for (v, w) in row.iter_mut().enumerate() {
            let mut c = 0.0;
            for e in elites {
                if e[u] == e[v] {
                    c += 1.0 / pf;
                }
            }
            *w = (1.0 - rho) * *w + rho * c;
        }
    });
}

impl Sim {
    fn decode(&self, g: &CsrGraph, genome: &Genome) -> (Labels, u32, u32) {
        match &self.dense {
            None => {
                let (l, s, r) = decode_counted(g, &self.edge, genome);
                (l, s as u32, r as u32)
            }
            Some(sm) => (dense_decode(g, sm, genome), 1, 0),
        }
    }

    fn encode(&self, g: &CsrGraph, labels: &Labels) -> Genome {
        match &self.dense {
            None => encode(g, &self.edge, labels),
            Some(sm) => dense_encode(sm, labels, g.n),
        }
    }

    fn update(&mut self, g: &CsrGraph, elites: &[&Labels], rho: f64) {
        match &mut self.dense {
            None => update_weights_floor(g, &mut self.edge, elites, rho, self.floor),
            Some(sm) => {
                dense_update(g.n, sm, elites, rho);
                self.edge = restrict(g, sm);
            }
        }
    }
}

#[derive(Clone)]
struct Mic {
    labels: Labels,
    obj: Obj,
    from_guidance: bool,
}

#[derive(Clone)]
struct Mac {
    genome: Genome,
    labels: Labels,
    obj: Obj,
}

fn keep<T: Clone>(pool: Vec<T>, objs: &[Obj], n: usize) -> (Vec<T>, Vec<usize>) {
    let idx = environment_selection(objs, n);
    let out = idx.iter().map(|&i| pool[i].clone()).collect();
    (out, idx)
}

fn four_objs(g: &CsrGraph, labels: &Labels) -> Obj {
    let mut v = evaluate(g, labels, ObjSet::HpIntraInter);
    v.extend(evaluate(g, labels, ObjSet::KkmRc));
    v
}

/// Rank-1 membership under an objective vector of any width. The population is
/// `2 N_p`, so the quadratic scan costs nothing and keeps the shipped
/// two-objective sweep untouched.
fn rank1_mask(objs: &[Obj]) -> Vec<bool> {
    let dominates = |a: &Obj, b: &Obj| {
        let mut strict = false;
        for (x, y) in a.iter().zip(b) {
            if x > y {
                return false;
            }
            if x < y {
                strict = true;
            }
        }
        strict
    };
    objs.par_iter()
        .map(|a| !objs.iter().any(|b| dominates(b, a)))
        .collect()
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run_probe(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize,
    p_c: f64,
    p_m: f64,
    gap: usize,
    do_refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
    abl: u32,
    sim_mode: u8,
    beta: f64,
    mac_mode: u8,
    front_mode: u8,
    seeds: &[Labels],
    w_floor: f64,
) -> (Vec<Labels>, Diag) {
    let t_start = Instant::now();
    let mut d = Diag::default();
    if g.n == 0 {
        return (vec![Vec::new()], d);
    }
    let gap = gap.max(1);
    let mode = SimMode::from_u8(sim_mode);
    let no_macro = abl & ABL_NO_MACRO != 0;
    let no_guidance = abl & ABL_NO_GUIDANCE != 0;
    let no_influence = abl & ABL_NO_INFLUENCE != 0;
    let no_w_update = abl & ABL_NO_W_UPDATE != 0 || mode == SimMode::KernelOnEdges;

    let micro_ops = MicroOps::from_topo(topo_mode);
    let cfg = Cfg::new(obj_mode);

    let mut sim = match mode {
        SimMode::Learned => Sim {
            edge: init_weights(g),
            dense: None,
            floor: w_floor,
        },
        SimMode::KernelOnEdges => {
            let sm = diffusion_kernel_csr(g, beta);
            Sim {
                edge: restrict(g, &sm),
                dense: None,
                floor: w_floor,
            }
        }
        SimMode::Dense => {
            let sm = diffusion_kernel_csr(g, beta);
            Sim {
                edge: restrict(g, &sm),
                dense: Some(sm),
                floor: w_floor,
            }
        }
    };

    d.cmax = macro_cmax(g.n, macro_cap) as u32;

    let mut seeded = init_micro_labels(g, pop);
    // External seeds replace the first slots of the initial micro population.
    // Elitist environmental selection keeps them only while nothing dominates
    // them, so a poor seed costs one slot for one generation.
    for (slot, seed) in seeded.iter_mut().zip(seeds.iter().take(pop)) {
        if seed.len() == g.n {
            slot.clone_from(seed);
        }
    }
    d.seeds_used = seeds.len().min(pop) as u32;
    let mut micro: Vec<Mic> = seeded
        .into_par_iter()
        .map(|labels| {
            let obj = cfg.eval_micro(g, &labels);
            Mic {
                labels,
                obj,
                from_guidance: false,
            }
        })
        .collect();

    let mut macro_pop: Vec<Mac> = if no_macro {
        Vec::new()
    } else {
        let genomes = init_macro_genomes(g, pop, macro_cap);
        for gn in &genomes {
            d.centres_init.push(centre_count(gn) as u32);
        }
        let decoded: Vec<(Labels, u32, u32)> =
            genomes.par_iter().map(|gn| sim.decode(g, gn)).collect();
        genomes
            .into_iter()
            .zip(decoded)
            .map(|(genome, (labels, s, r))| {
                d.sweeps.push(s);
                d.fallback_rounds.push(r);
                d.decode_calls += 1;
                let obj = cfg.eval_macro(g, &labels);
                Mac {
                    genome,
                    labels,
                    obj,
                }
            })
            .collect()
    };

    for t in 1..=num_gens {
        let t_mic = Instant::now();
        let mobjs: Vec<Obj> = micro.iter().map(|x| x.obj.clone()).collect();
        let mr = fast_nondominated_sort(&mobjs);
        let mc = crowding_distance(&mobjs, &mr);
        let mlabels: Vec<Labels> = micro.iter().map(|x| x.labels.clone()).collect();
        let raw = if micro_ops.any() {
            micro_offspring_topo(
                g,
                &sim.edge,
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
        };
        let micro_off: Vec<Mic> = raw
            .into_par_iter()
            .map(|l| {
                let obj = cfg.eval_micro(g, &l);
                Mic {
                    labels: l,
                    obj,
                    from_guidance: false,
                }
            })
            .collect();
        d.t_micro += t_mic.elapsed().as_secs_f64();

        let t_mac = Instant::now();
        let macro_off: Vec<Mac> = if no_macro {
            Vec::new()
        } else {
            let aobjs: Vec<Obj> = macro_pop.iter().map(|x| x.obj.clone()).collect();
            let ar = fast_nondominated_sort(&aobjs);
            let ac = crowding_distance(&aobjs, &ar);
            let agen: Vec<Genome> = macro_pop.iter().map(|x| x.genome.clone()).collect();
            let kids = macro_offspring_mode(&agen, &ar, &ac, p_m, 2 * t as u64 + 1, mac_mode);
            for gn in &kids {
                d.centres_off.push(centre_count(gn) as u32);
            }
            let decoded: Vec<(Labels, u32, u32)> =
                kids.par_iter().map(|gn| sim.decode(g, gn)).collect();
            kids.into_iter()
                .zip(decoded)
                .map(|(genome, (labels, s, r))| {
                    d.sweeps.push(s);
                    d.fallback_rounds.push(r);
                    d.decode_calls += 1;
                    let obj = cfg.eval_macro(g, &labels);
                    Mac {
                        genome,
                        labels,
                        obj,
                    }
                })
                .collect()
        };
        d.t_macro += t_mac.elapsed().as_secs_f64();

        if t % gap == 0 {
            let t_ex = Instant::now();
            if no_guidance || no_macro {
                micro.extend(micro_off);
                let objs: Vec<Obj> = micro.iter().map(|x| x.obj.clone()).collect();
                micro = keep(micro, &objs, pop).0;
            } else {
                let aobjs: Vec<Obj> = macro_pop.iter().map(|x| x.obj.clone()).collect();
                let ranks = fast_nondominated_sort(&aobjs);
                let elite_gen: Vec<&Genome> = macro_pop
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| ranks[*i] == 1)
                    .map(|(_, m)| &m.genome)
                    .collect();
                let decoded: Vec<(Labels, u32, u32)> =
                    elite_gen.par_iter().map(|gn| sim.decode(g, gn)).collect();
                let mut pool: Vec<Mic> = Vec::with_capacity(decoded.len());
                for (labels, s, r) in decoded {
                    d.sweeps.push(s);
                    d.fallback_rounds.push(r);
                    d.decode_calls += 1;
                    let obj = cfg.eval_micro(g, &labels);
                    pool.push(Mic {
                        labels,
                        obj,
                        from_guidance: true,
                    });
                }
                let injected = pool.len();
                pool.extend(micro);
                pool.extend(micro_off);
                let objs: Vec<Obj> = pool.iter().map(|x| x.obj.clone()).collect();
                let (next, idx) = keep(pool, &objs, pop);
                d.guidance_injected.push(injected as u32);
                d.guidance_survived
                    .push(idx.iter().filter(|&&i| i < injected).count() as u32);
                micro = next;
            }

            let mobjs: Vec<Obj> = micro.iter().map(|x| x.obj.clone()).collect();
            let ranks = fast_nondominated_sort(&mobjs);
            let elites: Vec<&Labels> = micro
                .iter()
                .enumerate()
                .filter(|(i, _)| ranks[*i] == 1)
                .map(|(_, m)| &m.labels)
                .collect();
            let rho = 0.5 * t as f64 / num_gens as f64;
            if !no_w_update {
                sim.update(g, &elites, rho);
            }

            if no_influence || no_macro {
                macro_pop.extend(macro_off);
                let objs: Vec<Obj> = macro_pop.iter().map(|x| x.obj.clone()).collect();
                macro_pop = keep(macro_pop, &objs, pop).0;
            } else {
                let made: Vec<(Genome, Labels, u32, u32)> = elites
                    .par_iter()
                    .map(|e| {
                        let genome = sim.encode(g, e);
                        let (labels, s, r) = sim.decode(g, &genome);
                        (genome, labels, s, r)
                    })
                    .collect();
                let mut pool: Vec<Mac> = Vec::with_capacity(made.len());
                for (genome, labels, s, r) in made {
                    d.centres_influence.push(centre_count(&genome) as u32);
                    d.sweeps.push(s);
                    d.fallback_rounds.push(r);
                    d.decode_calls += 1;
                    let obj = cfg.eval_macro(g, &labels);
                    pool.push(Mac {
                        genome,
                        labels,
                        obj,
                    });
                }
                let injected = pool.len();
                pool.extend(macro_pop);
                pool.extend(macro_off);
                let objs: Vec<Obj> = pool.iter().map(|x| x.obj.clone()).collect();
                let (next, idx) = keep(pool, &objs, pop);
                d.influence_injected.push(injected as u32);
                d.influence_survived
                    .push(idx.iter().filter(|&&i| i < injected).count() as u32);
                macro_pop = next;
            }
            d.t_exchange += t_ex.elapsed().as_secs_f64();
        } else {
            micro.extend(micro_off);
            let objs: Vec<Obj> = micro.iter().map(|x| x.obj.clone()).collect();
            micro = keep(micro, &objs, pop).0;
            if !no_macro {
                macro_pop.extend(macro_off);
                let objs: Vec<Obj> = macro_pop.iter().map(|x| x.obj.clone()).collect();
                macro_pop = keep(macro_pop, &objs, pop).0;
            }
        }
        for m in &macro_pop {
            d.centres_pop.push(centre_count(&m.genome) as u32);
        }
    }

    let t_post = Instant::now();
    let het = cfg.micro != cfg.macro_;
    let n_mic = micro.len();
    let mut labels: Vec<Labels> = Vec::with_capacity(n_mic + macro_pop.len());
    let mut objs: Vec<Obj> = Vec::with_capacity(n_mic + macro_pop.len());
    let mut guided: Vec<bool> = Vec::with_capacity(n_mic + macro_pop.len());
    for m in micro {
        labels.push(m.labels);
        objs.push(m.obj);
        guided.push(m.from_guidance);
    }
    for m in macro_pop {
        let obj = if het {
            cfg.eval_micro(g, &m.labels)
        } else {
            m.obj
        };
        labels.push(m.labels);
        objs.push(obj);
        guided.push(false);
    }

    let four: Vec<Obj> = labels.par_iter().map(|l| four_objs(g, l)).collect();
    let ranks = fast_nondominated_sort(&objs);
    let mask4 = rank1_mask(&four);
    for (i, &r) in ranks.iter().enumerate() {
        if r != 1 {
            continue;
        }
        if i < n_mic {
            d.front_from_micro += 1;
            if guided[i] {
                d.front_from_guidance += 1;
            }
        } else {
            d.front_from_macro += 1;
        }
    }
    d.front4_size = mask4.iter().filter(|&&r| r).count() as u32;
    d.front4_only = mask4
        .iter()
        .zip(&ranks)
        .filter(|&(&r4, &r2)| r4 && r2 != 1)
        .count() as u32;

    let deliver: Vec<bool> = if front_mode == 1 {
        mask4
    } else {
        ranks.iter().map(|&r| r == 1).collect()
    };
    let front: Vec<Labels> = labels
        .into_iter()
        .zip(deliver)
        .filter(|(_, k)| *k)
        .map(|(l, _)| l)
        .collect();
    let front = if front.is_empty() {
        vec![(0..g.n as i32).collect()]
    } else {
        front
    };
    d.front_size = front.len() as u32;

    let out = if do_refine {
        refine_front_mode(g, &sim.edge, front, cfg.micro, abl & ABL_NO_COARSEN == 0)
    } else {
        front
    };
    d.front_size_refined = out.len() as u32;
    if g.adj.len() <= 2_000_000 {
        d.w_final = sim.edge.clone();
    }
    d.t_post = t_post.elapsed().as_secs_f64();
    d.t_total = t_start.elapsed().as_secs_f64();
    (out, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::config::defaults::*;
    use crate::core::algorithms::smocc::macro_micro::run_fronts;
    use crate::core::algorithms::smocc::utils::fixtures::two_clique_edges;

    fn ring(k: i32, s: i32) -> CsrGraph {
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
        CsrGraph::from_edges(&nodes, &e)
    }

    #[test]
    fn probe_defaults_reproduce_the_shipped_front() {
        for g in [
            CsrGraph::from_edges(&(0..10).collect::<Vec<i32>>(), &two_clique_edges()),
            ring(6, 7),
        ] {
            let shipped = run_fronts(
                &g,
                40,
                30,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                true,
                DEFAULT_TOPO_MODE,
                DEFAULT_OBJ_MODE,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
            );
            let (probed, _) = run_probe(
                &g,
                40,
                30,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                true,
                DEFAULT_TOPO_MODE,
                DEFAULT_OBJ_MODE,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
                0,
                0,
                0.05,
                DEFAULT_MAC_MODE,
                0,
                &[],
                0.0,
            );
            assert_eq!(shipped, probed, "probe diverged from the shipped engine");
        }
    }

    #[test]
    fn ablation_switches_change_the_run_without_panicking() {
        let g = ring(6, 7);
        for abl in [
            ABL_NO_MACRO,
            ABL_NO_GUIDANCE,
            ABL_NO_INFLUENCE,
            ABL_NO_W_UPDATE,
        ] {
            let (front, d) = run_probe(
                &g,
                30,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                true,
                DEFAULT_TOPO_MODE,
                DEFAULT_OBJ_MODE,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
                abl,
                0,
                0.05,
                0,
                0,
                &[],
                0.0,
            );
            assert!(!front.is_empty());
            assert!(front.iter().all(|p| p.len() == g.n));
            if abl == ABL_NO_MACRO {
                assert_eq!(d.front_from_macro, 0);
                assert!(d.centres_init.is_empty());
            }
        }
    }

    #[test]
    fn a_seed_survives_when_nothing_dominates_it() {
        let g = ring(6, 7);
        let planted: Labels = (0..g.n as i32).map(|i| i / 7).collect();
        let (front, d) = run_probe(
            &g,
            30,
            5,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            true,
            DEFAULT_TOPO_MODE,
            DEFAULT_OBJ_MODE,
            DEFAULT_MACRO_CAP,
            DEFAULT_MICRO_MUT,
            0,
            0,
            0.05,
            DEFAULT_MAC_MODE,
            0,
            std::slice::from_ref(&planted),
            0.0,
        );
        assert_eq!(d.seeds_used, 1);
        let best = front
            .iter()
            .map(|p| {
                let mut u = p.clone();
                u.sort_unstable();
                u.dedup();
                u.len()
            })
            .min()
            .unwrap();
        assert!(best <= 6, "the planted seed left no coarse member: {best}");
    }

    #[test]
    fn dense_modes_run_and_report_sweeps() {
        let g = ring(5, 6);
        for sim in [1u8, 2] {
            let (front, d) = run_probe(
                &g,
                20,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                true,
                DEFAULT_TOPO_MODE,
                DEFAULT_OBJ_MODE,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
                0,
                sim,
                0.05,
                0,
                0,
                &[],
                0.0,
            );
            assert!(!front.is_empty());
            assert!(d.decode_calls > 0);
            assert_eq!(d.sweeps.len() as u64, d.decode_calls);
        }
    }
}
