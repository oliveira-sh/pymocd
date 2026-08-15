//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::distr::Bernoulli;
use rand::{Rng, RngExt};
use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use crate::core::algorithms::smocc2::mopso::pareto::dominates;
use crate::core::algorithms::smocc2::sim::decode;
use crate::core::algorithms::smocc2::{Genome, Labels};
use crate::core::algorithms::smocc2::config::defaults::{C1, C2};
use crate::core::algorithms::smocc2::config::objectives::Cfg;
use crate::core::algorithms::smocc2::gpu::Gpu;

use super::particles::{MacElite, MacParticle, MicElite, MicParticle};

fn bernoulli(p: f64) -> Bernoulli {
    match Bernoulli::new(p) {
        Ok(d) => d,
        Err(_) => panic!("p={p:?} is outside range [0.0, 1.0]"),
    }
}

fn leader_tournament(crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = crowd.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    if crowd[i] >= crowd[j] { i } else { j }
}

fn pbest_wants_new(new: &[f64], pbest: &[f64], r: &mut impl Rng) -> bool {
    if dominates(new, pbest) {
        true
    } else if dominates(pbest, new) {
        false
    } else {
        r.random_bool(0.5)
    }
}

fn micro_move(
    g: &CsrGraph,
    p: &mut MicParticle,
    arch: &[MicElite],
    crowd: &[f64],
    w: f64,
    d_turb: &Bernoulli,
) {
    let n = g.n;
    {
        let mut r = rand::rng();
        let leader = &arch[leader_tournament(crowd, &mut r)].labels;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let d1 = (p.pbest[i] != p.x[i]) as u8 as f64;
            let d2 = (leader[i] != p.x[i]) as u8 as f64;
            let r1: f64 = r.random();
            let r2: f64 = r.random();
            p.v[i] = (w * p.v[i] + C1 * r1 * d1 + C2 * r2 * d2).clamp(0.0, 1.0);
        }

        let mut freq: Vec<u32> = vec![0u32; n];
        let mut touched: Vec<usize> = Vec::new();
        for i in 0..n {
            if r.random::<f64>() >= p.v[i] {
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

        for i in 0..n {
            let nbrs = g.neighbors(i);
            if !nbrs.is_empty() && r.sample(*d_turb) {
                let j = nbrs[r.random_range(0..nbrs.len())] as usize;
                p.x[i] = p.x[j];
            }
        }
    }
}

fn micro_finish(g: &CsrGraph, p: &mut MicParticle, obj: Option<crate::core::algorithms::smocc2::mopso::pareto::Obj>, cfg: &Cfg) {
    p.obj = match obj {
        Some(o) => o,
        None => cfg.eval_micro(g, &p.x),
    };
    let mut r = rand::rng();
    if pbest_wants_new(&p.obj, &p.pbest_obj, &mut r) {
        p.pbest.clone_from(&p.x);
        p.pbest_obj.clone_from(&p.obj);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn micro_step(
    g: &CsrGraph,
    parts: &mut [MicParticle],
    arch: &[MicElite],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
) {
    if arch.is_empty() {
        return;
    }
    let d_turb = bernoulli(p_t);
    parts.par_iter_mut().for_each(|p| {
        micro_move(g, p, arch, crowd, w, &d_turb);
        micro_finish(g, p, None, cfg);
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn micro_step_gpu(
    gpu: &mut Gpu,
    parts: &mut [MicParticle],
    arch: &[MicElite],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
) {
    if arch.is_empty() {
        return;
    }
    let arch_labels: Vec<&Labels> = arch.iter().map(|a| &a.labels).collect();
    gpu.upload_arch(&arch_labels)
        .expect("CUDA runtime failure in archive upload");
    let mut r = rand::rng();
    let leaders: Vec<i32> = (0..parts.len())
        .map(|_| leader_tournament(crowd, &mut r) as i32)
        .collect();
    let seed: u64 = r.random();
    let objs = gpu
        .micro_step(parts.len(), &leaders, w, C1, C2, p_t, seed)
        .expect("CUDA runtime failure in micro step");
    let mut accepted: Vec<usize> = Vec::new();
    for (i, (p, o)) in parts.iter_mut().zip(&objs).enumerate() {
        p.obj = cfg.pick_micro(o);
        if pbest_wants_new(&p.obj, &p.pbest_obj, &mut r) {
            p.pbest_obj.clone_from(&p.obj);
            accepted.push(i);
        }
    }
    gpu.micro_pbest_commit(&accepted)
        .expect("CUDA runtime failure in pbest commit");
    let xs = gpu
        .micro_download(parts.len())
        .expect("CUDA runtime failure in micro download");
    for (p, x) in parts.iter_mut().zip(xs) {
        p.x = x;
    }
}

fn macro_move(n: usize, p: &mut MacParticle, arch: &[MacElite], crowd: &[f64], w: f64, p_t: f64) {
    let mut r = rand::rng();
    let leader = &arch[leader_tournament(crowd, &mut r)].genome;

    let mut add: Vec<(f64, u32)> = Vec::new();
    let mut del: Vec<(f64, u32)> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let s = p.genome[i] != 0;
        let pb = p.pbest[i] != 0;
        let ld = leader[i] != 0;
        if !s && (pb || ld) {
            let mut wt = 0.0;
            if pb {
                wt += C1 * r.random::<f64>();
            }
            if ld {
                wt += C2 * r.random::<f64>();
            }
            add.push((wt, i as u32));
        } else if s && !(pb && ld) {
            let mut wt = 0.0;
            if !pb {
                wt += C1 * r.random::<f64>();
            }
            if !ld {
                wt += C2 * r.random::<f64>();
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

    if r.random_bool(p_t) {
        let centers: Vec<usize> = (0..n).filter(|&i| p.genome[i] != 0).collect();
        let c = centers.len();
        if c > 0 && c < n {
            let out = centers[r.random_range(0..c)];
            loop {
                let cand = r.random_range(0..n);
                if p.genome[cand] == 0 {
                    p.genome[out] = 0;
                    p.genome[cand] = 1;
                    break;
                }
            }
        }
    }
}

fn macro_finish(g: &CsrGraph, p: &mut MacParticle, labels: Labels, cfg: &Cfg) {
    p.labels = labels;
    p.obj = cfg.eval_macro(g, &p.labels);
    let mut r = rand::rng();
    if pbest_wants_new(&p.obj, &p.pbest_obj, &mut r) {
        p.pbest.clone_from(&p.genome);
        p.pbest_obj.clone_from(&p.obj);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn macro_step(
    g: &CsrGraph,
    wadj: &[f64],
    parts: &mut [MacParticle],
    arch: &[MacElite],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
) {
    if arch.is_empty() {
        return;
    }
    let n = g.n;
    parts.par_iter_mut().for_each(|p| {
        macro_move(n, p, arch, crowd, w, p_t);
        let labels = decode(g, wadj, &p.genome);
        macro_finish(g, p, labels, cfg);
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn macro_step_gpu(
    g: &CsrGraph,
    gpu: &mut Gpu,
    parts: &mut [MacParticle],
    arch: &[MacElite],
    crowd: &[f64],
    cfg: &Cfg,
    w: f64,
    p_t: f64,
) {
    if arch.is_empty() {
        return;
    }
    let n = g.n;
    parts
        .par_iter_mut()
        .for_each(|p| macro_move(n, p, arch, crowd, w, p_t));
    let genomes: Vec<&Genome> = parts.iter().map(|p| &p.genome).collect();
    let (labels, objs) = gpu
        .decode_eval(g, &genomes)
        .expect("CUDA runtime failure in macro decode");
    for (p, (l, o)) in parts.iter_mut().zip(labels.into_iter().zip(objs)) {
        p.labels = l;
        p.obj = cfg.pick_macro(&o);
        let mut r = rand::rng();
        if pbest_wants_new(&p.obj, &p.pbest_obj, &mut r) {
            p.pbest.clone_from(&p.genome);
            p.pbest_obj.clone_from(&p.obj);
        }
    }
}
