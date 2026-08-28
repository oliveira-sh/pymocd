//! Algorithm 1: the generational loop over both swarms and the final mergence.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mmcomo::nsga2::fast_nondominated_sort;
use crate::core::algorithms::mmcomo::objectives::kkm_rc;
use crate::core::algorithms::mmcomo::operators::{local_search, macro_offspring, micro_offspring};
use crate::core::algorithms::mmcomo::similarity::{decode, diffusion_kernel};
use crate::core::algorithms::mmcomo::{Genome, Graph, Labels, Sm};

use super::exchange::{guidance, influence};
use super::init::{init_macro, init_micro};
use super::swarms::{Mac, Mic, macro_objs, micro_objs, ranks_and_crowd, select_macro, select_micro};

/// Micro offspring (Alg. 1 line 7), evaluated.
fn breed_micro(g: &Graph, micro: &[Mic], p_c: f64) -> Vec<Mic> {
    let (ranks, crowd) = ranks_and_crowd(&micro_objs(micro));
    let parents: Vec<Labels> = micro.iter().map(|x| x.labels.clone()).collect();
    micro_offspring(g, &parents, &ranks, &crowd, p_c)
        .into_iter()
        .map(|labels| {
            let obj = kkm_rc(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

/// Macro offspring (Alg. 1 line 5), decoded and evaluated.
fn breed_macro(g: &Graph, sm: &Sm, macro_pop: &[Mac], p_m: f64) -> Vec<Mac> {
    let (ranks, crowd) = ranks_and_crowd(&macro_objs(macro_pop));
    let parents: Vec<Genome> = macro_pop.iter().map(|x| x.genome.clone()).collect();
    macro_offspring(&parents, &ranks, &crowd, p_m)
        .into_iter()
        .map(|genome| {
            let labels = decode(g, sm, &genome);
            let obj = kkm_rc(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
        })
        .collect()
}

/// Modularity local search (Alg. 1 line 11, ref [38]) on rank-1 micro members.
fn local_search_front(g: &Graph, micro: &mut [Mic]) {
    let ranks = fast_nondominated_sort(&micro_objs(micro));
    for (i, m) in micro.iter_mut().enumerate() {
        if ranks[i] == 1 {
            local_search(g, &mut m.labels);
            m.obj = kkm_rc(g, &m.labels);
        }
    }
}

/// Mergence (Alg. 1 Phase 3): rank-1 of micro ∪ macro, all-singletons if empty.
fn mergence(micro: Vec<Mic>, macro_pop: Vec<Mac>, n: usize) -> Vec<Labels> {
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
    if front.is_empty() {
        vec![(0..n as i32).collect()]
    } else {
        front
    }
}

/// Algorithm 1. Returns the rank-1 front of the merged populations.
pub fn run_fronts(
    g: &Graph,
    pop: usize,
    num_gens: usize,
    p_c: f64,
    p_m: f64,
    gap: usize,
    beta: f64,
) -> Vec<Labels> {
    if g.n == 0 {
        return vec![Vec::new()];
    }
    let gap = gap.max(1);
    let mut sm = diffusion_kernel(g, beta);
    let mut micro = init_micro(g, pop);
    let mut macro_pop = init_macro(g, &sm, pop);

    for t in 1..=num_gens {
        let micro_off = breed_micro(g, &micro, p_c);
        let macro_off = breed_macro(g, &sm, &macro_pop, p_m);

        if t % gap == 0 {
            micro = guidance(g, &sm, &macro_pop, micro, micro_off, pop);
            local_search_front(g, &mut micro);
            macro_pop = influence(g, &mut sm, &micro, macro_pop, macro_off, t, num_gens, pop);
        } else {
            micro.extend(micro_off);
            micro = select_micro(micro, pop);
            macro_pop.extend(macro_off);
            macro_pop = select_macro(macro_pop, pop);
        }
    }

    mergence(micro, macro_pop, g.n)
}
