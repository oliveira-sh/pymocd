//! The SMOCC generational loop driving the macro and micro populations and
//! the final non-dominated front extraction.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::smocc::config::{Cfg, MicroOps};
use crate::core::algorithms::smocc::front::refine_front;
use crate::core::algorithms::smocc::nsga2::{Obj, crowding_distance, fast_nondominated_sort};
use crate::core::algorithms::smocc::operators::{
    macro_offspring, micro_offspring, micro_offspring_topo,
};
use crate::core::algorithms::smocc::similarity::decode;
use crate::core::algorithms::smocc::similarity::init_weights;
use crate::core::algorithms::smocc::{Genome, Labels};
use crate::core::graph::CsrGraph;

use super::exchange::{guidance, influence};
use super::init::{init_macro, init_micro};
use super::swarms::{Mac, Mic, macro_objs, micro_objs, select_macro, select_micro};

fn ranks_and_crowd(objs: &[Obj]) -> (Vec<usize>, Vec<f64>) {
    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);
    (ranks, crowd)
}

#[allow(clippy::too_many_arguments)]
pub fn run_fronts(
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
                &wadj,
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
        refine_front(g, &wadj, front, cfg.micro)
    } else {
        front
    }
}
