//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>

use crate::core::graph::CsrGraph;

use super::config::defaults::DEFAULT_OBJ_MODE;
use super::gpu::Gpu;
use super::mopso::engine::run_fronts;
use super::select::{select_best, to_output};

#[allow(clippy::too_many_arguments)]
pub fn gmocs(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    macro_cap: f64,
) -> Result<Vec<(i32, i32)>, String> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Ok(Vec::new());
    }
    let mut dev = Gpu::new(&g, pop)?;
    let front = run_fronts(
        &g,
        pop,
        num_gens,
        gap,
        turb,
        true,
        DEFAULT_OBJ_MODE,
        macro_cap,
        &mut dev,
    );
    let best = select_best(&g, front, &mut dev, pop);
    Ok(to_output(&g, &best))
}

#[allow(clippy::too_many_arguments)]
pub fn gmocs_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    refine: bool,
    obj_mode: u16,
    macro_cap: f64,
) -> Result<Vec<Vec<(i32, i32)>>, String> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Ok(Vec::new());
    }
    let mut dev = Gpu::new(&g, pop)?;
    Ok(run_fronts(
        &g,
        pop,
        num_gens,
        gap,
        turb,
        refine,
        obj_mode,
        macro_cap,
        &mut dev,
    )
    .iter()
    .map(|l| to_output(&g, l))
    .collect())
}
