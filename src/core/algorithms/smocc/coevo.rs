//! Co-evolutionary exchange between the two swarms: macro elites guide the
//! micro population, micro elites reshape the similarity weights and seed macro.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use super::Labels;
use super::config::objectives::Cfg;
use super::nsga2::fast_nondominated_sort;
use super::particles::{Mac, Mic, macro_objs, micro_objs, select_macro, select_micro};
use super::sim::{decode, encode, update_weights};

pub(super) fn guidance(
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
pub(super) fn influence(
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
