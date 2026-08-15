use crate::core::graph::CsrGraph;

use super::super::smocc::{select_best, to_output};
use super::defaults::DEFAULT_OBJ_MODE;
use super::engine::run_fronts;

#[allow(clippy::too_many_arguments)]
pub fn smocc2(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    gap: usize,
    turb: f64,
    macro_cap: f64,
    gpu: bool,
) -> Result<Vec<(i32, i32)>, String> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Ok(Vec::new());
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
        gpu,
    )?;
    let best = select_best(&g, front);
    Ok(to_output(&g, &best))
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
    gpu: bool,
) -> Result<Vec<Vec<(i32, i32)>>, String> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Ok(Vec::new());
    }
    Ok(
        run_fronts(&g, pop, num_gens, gap, turb, refine, obj_mode, macro_cap, gpu)?
            .iter()
            .map(|l| to_output(&g, l))
            .collect(),
    )
}
