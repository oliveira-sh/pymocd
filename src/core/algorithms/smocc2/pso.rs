//! PSO update operators for SMOCC-II. These replace SMOCC's GA operators:
//! there is no crossover, mutation or tournament over offspring — each swarm
//! member is a particle moved toward its personal best and an archive leader.
use rand::distr::Bernoulli;
use rand::rngs::StdRng;
use rand::{Rng, RngExt};
use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use super::super::smocc::nsga2::dominates;
use super::super::smocc::operators::slot_rng;
use super::super::smocc::sim::decode;
use super::super::smocc::Labels;
use super::defaults::{C1, C2};
use super::{Cfg, MacA, MacP, MicA, MicP};

// One RNG per particle per phase per generation: the salt encodes (t, phase)
// and `slot_rng` folds in the particle index, so every stochastic draw is a
// pure function of (t, phase, k) — the hard determinism requirement. Init
// reuses SMOCC's salts (u64::MAX, u64::MAX - 1), far outside this range.
const PHASES: u64 = 10;
const PH_MIC_LEAD: u64 = 0;
const PH_MIC_VEL: u64 = 1;
const PH_MIC_POS: u64 = 2;
const PH_MIC_TURB: u64 = 3;
const PH_MIC_PB: u64 = 4;
const PH_MAC_LEAD: u64 = 5;
const PH_MAC_UPD: u64 = 6;
const PH_MAC_TURB: u64 = 7;
const PH_MAC_PB: u64 = 8;

fn phase_rng(t: usize, phase: u64, k: usize) -> StdRng {
    slot_rng((t as u64) * PHASES + phase, k)
}

#[inline]
fn bernoulli(p: f64) -> Bernoulli {
    match Bernoulli::new(p) {
        Ok(d) => d,
        Err(_) => panic!("p={p:?} is outside range [0.0, 1.0]"),
    }
}

/// Binary tournament on crowding distance among archive members: the less
/// crowded member leads. Ties keep the first pick, as SMOCC's tournament does.
#[inline]
fn leader_tournament(crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = crowd.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    if crowd[i] >= crowd[j] { i } else { j }
}

/// pbest replacement rule: new position replaces pbest when it dominates it,
/// never when dominated, and with probability 0.5 when mutually nondominated
/// (exact objective ties included).
#[inline]
fn pbest_wants_new(new: &[f64], pbest: &[f64], r: &mut StdRng) -> bool {
    if dominates(new, pbest) {
        true
    } else if dominates(pbest, new) {
        false
    } else {
        r.random_bool(0.5)
    }
}

/// One micro-swarm PSO step: velocity, attraction toward the leader via
/// neighbour majority, turbulence, evaluation, pbest update. Label choices
/// only ever copy node ids already present in a leader or the particle, so
/// the dense-slot objective invariant (labels in [0, n)) is preserved.
#[allow(clippy::too_many_arguments)]
pub(super) fn micro_step(
    g: &CsrGraph,
    parts: &mut [MicP],
    arch: &[MicA],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
    t: usize,
) {
    let n = g.n;
    if arch.is_empty() {
        return;
    }
    let d_turb = bernoulli(p_t);
    parts.par_iter_mut().enumerate().for_each(|(k, p)| {
        let mut rl = phase_rng(t, PH_MIC_LEAD, k);
        let leader = &arch[leader_tournament(crowd, &mut rl)].labels;

        // Velocity: elementwise disagreement pulls, inertia decays.
        let mut rv = phase_rng(t, PH_MIC_VEL, k);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let d1 = (p.pb[i] != p.x[i]) as u8 as f64;
            let d2 = (leader[i] != p.x[i]) as u8 as f64;
            let r1: f64 = rv.random();
            let r2: f64 = rv.random();
            p.v[i] = (w * p.v[i] + C1 * r1 * d1 + C2 * r2 * d2).clamp(0.0, 1.0);
        }

        // Position: with probability V[i], adopt the majority label of i's
        // graph neighbours under the LEADER's partition (topology-feasible
        // attraction); leader[i] on tie or when i has no neighbours.
        let mut rp = phase_rng(t, PH_MIC_POS, k);
        let mut freq: Vec<u32> = vec![0u32; n];
        let mut touched: Vec<usize> = Vec::new();
        for i in 0..n {
            if rp.random::<f64>() >= p.v[i] {
                continue;
            }
            touched.clear();
            let mut max_c = 0u32;
            for &vtx in g.neighbors(i) {
                let li = leader[vtx as usize] as usize;
                let e = &mut freq[li];
                if *e == 0 {
                    touched.push(li);
                }
                *e += 1;
                if *e > max_c {
                    max_c = *e;
                }
            }
            let mut winner = leader[i];
            let mut n_max = 0u32;
            for &s in &touched {
                if freq[s] == max_c {
                    n_max += 1;
                    winner = s as i32;
                }
            }
            p.x[i] = if n_max == 1 { winner } else { leader[i] };
            for &s in &touched {
                freq[s] = 0;
            }
        }

        // Turbulence: without it a mutation-free discrete swarm collapses.
        // In-place sequential neighbour-copy, exactly SMOCC's mutation shape.
        let mut rt = phase_rng(t, PH_MIC_TURB, k);
        for i in 0..n {
            let nbrs = g.neighbors(i);
            if !nbrs.is_empty() && rt.sample(d_turb) {
                let j = nbrs[rt.random_range(0..nbrs.len())] as usize;
                p.x[i] = p.x[j];
            }
        }

        p.obj = cfg.eval_micro(g, &p.x);

        let mut rb = phase_rng(t, PH_MIC_PB, k);
        if pbest_wants_new(&p.obj, &p.pb_obj, &mut rb) {
            p.pb.clone_from(&p.x);
            p.pb_obj.clone_from(&p.obj);
        }
    });
}

/// Move phase of one macro particle: leader tournament, the set-based swap
/// update, and turbulence — everything except decode/eval/pbest, so the CPU
/// and GPU paths share it (and its RNG phase streams) exactly.
fn macro_move_one(
    n: usize,
    k: usize,
    p: &mut MacP,
    arch: &[MacA],
    crowd: &[f64],
    w: f64,
    p_t: f64,
    t: usize,
) {
    {
        let mut rl = phase_rng(t, PH_MAC_LEAD, k);
        let leader = &arch[leader_tournament(crowd, &mut rl)].genome;

        // ADD = (pbest ∪ leader) \ S, DEL = S \ (pbest ∩ leader); each element
        // scored by which best contributed it. Taking the top-k of fresh
        // c·r-weighted scores IS the weighted sample: elements backed by both
        // bests are stochastically favoured, none is ever certain.
        let mut ru = phase_rng(t, PH_MAC_UPD, k);
        let mut add: Vec<(f64, u32)> = Vec::new();
        let mut del: Vec<(f64, u32)> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let s = p.genome[i] != 0;
            let pb = p.pb[i] != 0;
            let ld = leader[i] != 0;
            if !s && (pb || ld) {
                let mut wt = 0.0;
                if pb {
                    wt += C1 * ru.random::<f64>();
                }
                if ld {
                    wt += C2 * ru.random::<f64>();
                }
                add.push((wt, i as u32));
            } else if s && !(pb && ld) {
                let mut wt = 0.0;
                if !pb {
                    wt += C1 * ru.random::<f64>();
                }
                if !ld {
                    wt += C2 * ru.random::<f64>();
                }
                del.push((wt, i as u32));
            }
        }
        let k_swaps = ((del.len() as f64) * (1.0 - w)).ceil() as usize;
        let k_swaps = k_swaps.min(del.len()).min(add.len());
        if k_swaps > 0 {
            let by_score = |a: &(f64, u32), b: &(f64, u32)| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            };
            add.sort_unstable_by(by_score);
            del.sort_unstable_by(by_score);
            for j in 0..k_swaps {
                p.genome[del[j].1 as usize] = 0;
                p.genome[add[j].1 as usize] = 1;
            }
        }

    }

    // Turbulence: swap one centre for a uniformly random non-centre node.
    // Swapping preserves |S|, so the ≥1-centre and cmax invariants from
    // init hold for the particle's whole lifetime.
    let mut rt = phase_rng(t, PH_MAC_TURB, k);
    if rt.random_bool(p_t) {
        let centers: Vec<usize> = (0..n).filter(|&i| p.genome[i] != 0).collect();
        let c = centers.len();
        if c > 0 && c < n {
            let out = centers[rt.random_range(0..c)];
            loop {
                let cand = rt.random_range(0..n);
                if p.genome[cand] == 0 {
                    p.genome[out] = 0;
                    p.genome[cand] = 1;
                    break;
                }
            }
        }
    }
}

/// Eval-and-pbest phase of one macro particle, given its decoded labels.
fn macro_finish_one(g: &CsrGraph, k: usize, p: &mut MacP, labels: Labels, cfg: &Cfg, t: usize) {
    p.labels = labels;
    p.obj = cfg.eval_macro(g, &p.labels);
    let mut rb = phase_rng(t, PH_MAC_PB, k);
    if pbest_wants_new(&p.obj, &p.pb_obj, &mut rb) {
        p.pb.clone_from(&p.genome);
        p.pb_obj.clone_from(&p.obj);
    }
}

/// One macro-swarm step: cardinality-preserving set-based binary PSO. The
/// velocity is a pair of scored candidate pools (ADD/DEL) and the inertia
/// weight sets the fraction of the disagreement resolved per step
/// (V_scale = 1 - w: each candidate swap is skipped with probability w).
#[allow(clippy::too_many_arguments)]
pub(super) fn macro_step(
    g: &CsrGraph,
    wadj: &[f64],
    parts: &mut [MacP],
    arch: &[MacA],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
    t: usize,
) {
    if arch.is_empty() {
        return;
    }
    let n = g.n;
    parts.par_iter_mut().enumerate().for_each(|(k, p)| {
        macro_move_one(n, k, p, arch, crowd, w, p_t, t);
        // Decode/eval is the hot path, exactly as in SMOCC.
        let labels = decode(g, wadj, &p.genome);
        macro_finish_one(g, k, p, labels, cfg, t);
    });
}

/// GPU variant of [`macro_step`]: identical move and eval/pbest phases (same
/// RNG streams), with the pop decodes batched into one device launch.
#[allow(clippy::too_many_arguments)]
pub(super) fn macro_step_gpu(
    g: &CsrGraph,
    gpu: &mut super::gpu::Gpu,
    parts: &mut [MacP],
    arch: &[MacA],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
    t: usize,
) {
    if arch.is_empty() {
        return;
    }
    let n = g.n;
    parts
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, p)| macro_move_one(n, k, p, arch, crowd, w, p_t, t));
    let genomes: Vec<&super::Genome> = parts.iter().map(|p| &p.genome).collect();
    let labels = gpu
        .batch_decode(g, &genomes)
        .expect("CUDA runtime failure in macro decode");
    parts
        .par_iter_mut()
        .zip(labels)
        .enumerate()
        .for_each(|(k, (p, l))| macro_finish_one(g, k, p, l, cfg, t));
}
