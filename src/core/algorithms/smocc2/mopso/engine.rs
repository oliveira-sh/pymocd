//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use crate::core::algorithms::smocc2::Labels;
use crate::core::algorithms::smocc2::mopso::pareto::{Obj, fast_nondominated_sort};
use crate::core::algorithms::smocc2::refine::refine_front;
use crate::core::algorithms::smocc2::sim::init_weights;
use crate::core::algorithms::smocc2::config::defaults::DEFAULT_TURB;
use crate::core::algorithms::smocc2::config::objectives::Cfg;
use crate::core::algorithms::smocc2::config::schedule::{inertia, turbulence};
use crate::core::algorithms::smocc2::gpu::Gpu;

use super::archive::{arch_crowd, update_macro_archive, update_micro_archive};
use super::coevo::{guidance, influence};
use super::init::{init_macro_swarm, init_micro_swarm};
use super::particles::{MacElite, MicElite};
use super::steps;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fronts(
    g: &CsrGraph,
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    do_refine: bool,
    obj_mode: u16,
    macro_cap: f64,
    dev: &mut Gpu,
) -> Result<Vec<Labels>, String> {
    if g.n == 0 {
        return Ok(vec![Vec::new()]);
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
    {
        let xs: Vec<&Labels> = mic.iter().map(|p| &p.x).collect();
        dev.micro_init(&xs)
            .expect("CUDA runtime failure in micro init");
    }
    let mut mac = init_macro_swarm(g, pop, &cfg, macro_cap, dev);

    let mut mic_arch = update_micro_archive(
        Vec::new(),
        mic.iter()
            .map(|p| MicElite {
                labels: p.x.clone(),
                obj: p.obj.clone(),
            })
            .collect(),
        pop,
    );
    let mut mac_arch = update_macro_archive(
        Vec::new(),
        mac.iter()
            .map(|p| MacElite {
                genome: p.genome.clone(),
                labels: p.labels.clone(),
                obj: p.obj.clone(),
            })
            .collect(),
        pop,
    );

    for t in 1..=num_gens {
        let w = inertia(t, num_gens);
        let p_t = turbulence(turb, w);

        let mic_objs: Vec<Obj> = mic_arch.iter().map(|a| a.obj.clone()).collect();
        let crowd = arch_crowd(&mic_objs);
        steps::micro_step(dev, &mut mic, &mic_arch, &crowd, &cfg, w, p_t);
        mic_arch = update_micro_archive(
            mic_arch,
            mic.iter()
                .map(|p| MicElite {
                    labels: p.x.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );

        let mac_objs: Vec<Obj> = mac_arch.iter().map(|a| a.obj.clone()).collect();
        let crowd = arch_crowd(&mac_objs);
        steps::macro_step(g, dev, &mut mac, &mac_arch, &crowd, &cfg, w, p_t);
        mac_arch = update_macro_archive(
            mac_arch,
            mac.iter()
                .map(|p| MacElite {
                    genome: p.genome.clone(),
                    labels: p.labels.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );

        if t % gap == 0 {
            mic_arch = guidance(
                g,
                &mac_arch,
                mic_arch,
                &mut mic,
                pop,
                &cfg,
                dev,
            );
            mac_arch = influence(
                g,
                &mut wadj,
                &mic_arch,
                mac_arch,
                t,
                num_gens,
                pop,
                &cfg,
                dev,
            );
        }
    }

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
    let mac_labels: Vec<Labels> = mac.iter().map(|p| p.labels.clone()).collect();
    let mac_objs_own: Vec<Obj> = mac.into_iter().map(|p| p.obj).collect();
    let arch_labels: Vec<Labels> = mac_arch.iter().map(|a| a.labels.clone()).collect();
    let arch_objs_own: Vec<Obj> = mac_arch.into_iter().map(|a| a.obj).collect();
    let mut mac_all: Vec<Labels> = mac_labels;
    mac_all.extend(arch_labels);
    let mut mac_all_objs: Vec<Obj> = mac_objs_own;
    mac_all_objs.extend(arch_objs_own);
    if het {
        let mut het_objs: Vec<Obj> = Vec::with_capacity(mac_all.len());
        for chunk in mac_all.chunks(pop.max(1)) {
            let refs: Vec<&Labels> = chunk.iter().collect();
            het_objs.extend(
                dev.eval_labels(&refs)
                    .expect("CUDA runtime failure in final eval")
                    .into_iter()
                    .map(|o| cfg.pick_micro(&o)),
            );
        }
        mac_all_objs = het_objs;
    }
    labels.extend(mac_all);
    objs.extend(mac_all_objs);
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

    Ok(if do_refine {
        refine_front(g, &wadj, front, cfg.micro)
    } else {
        front
    })
}
