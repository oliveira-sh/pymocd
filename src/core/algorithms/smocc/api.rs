//! Public entry points for SMOCC: the Pareto-front search and the
//! single-partition wrappers re-exported from the crate root.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::config::defaults::{
    DEFAULT_MACRO_CAP, DEFAULT_MICRO_MUT, DEFAULT_OBJ_MODE, DEFAULT_TOPO_MODE,
};
use super::engine::run_fronts;
use super::select::{select_best, to_output};

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
