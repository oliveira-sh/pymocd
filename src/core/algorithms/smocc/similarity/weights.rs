//! Unit initialisation of the edge weights and the elite-consensus
//! reinforcement that closes the co-evolutionary loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::config::defaults::DEFAULT_W_FLOOR;
use crate::core::graph::CsrGraph;

pub fn init_weights(g: &CsrGraph) -> Vec<f64> {
    vec![1.0f64; g.adj.len()]
}

pub fn update_weights(g: &CsrGraph, wadj: &mut [f64], elites: &[&Labels], rho: f64) {
    update_weights_floor(g, wadj, elites, rho, DEFAULT_W_FLOOR);
}

/// Smallest weight the consensus update may leave on an edge.
///
/// The update has an absorbing state on paper: an edge every elite cuts takes
/// `c = 0`, so its weight decays geometrically, and once it is near zero the
/// decoder cannot propagate a label across it, no later elite can join its
/// endpoints, and `c` stays zero forever. A floor would keep the merge
/// reachable.
///
/// Measured, that state is never reached, so the floor is inert. On LFR graphs
/// at `n = 10^4` the smallest learned weight is `0.096` at a mixing of `0.5`
/// and `0.110` at `0.6`, and not one edge of `294_000` falls below `0.05`; the
/// elite set spans granularity, so a coarse elite keeps almost every pair
/// together and `c` is bounded away from zero. Floors at `0.01`, `0.05` and
/// `0.10` reproduce the shipped run exactly. The parameter is kept because the
/// paper reports that measurement, not because it changes anything.
pub fn update_weights_floor(
    g: &CsrGraph,
    wadj: &mut [f64],
    elites: &[&Labels],
    rho: f64,
    floor: f64,
) {
    let two_m = g.adj.len();
    if two_m == 0 || elites.is_empty() {
        return;
    }
    let pf = elites.len() as f64;
    let cov: Vec<f64> = (0..g.n)
        .into_par_iter()
        .flat_map_iter(|u| {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            (start..end).map(move |p| {
                let v = g.adj[p] as usize;
                let mut c = 0.0;
                for e in elites {
                    if e[u] == e[v] {
                        c += 1.0 / pf;
                    }
                }
                c
            })
        })
        .collect();
    let floor = floor.clamp(0.0, 1.0);
    wadj.par_iter_mut().zip(cov.par_iter()).for_each(|(w, &c)| {
        *w = ((1.0 - rho) * *w + rho * c).max(floor);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::utils::fixtures::two_triangles;

    #[test]
    fn update_weights_raises_intra_lowers_inter() {
        let g = two_triangles();
        let mut w = init_weights(&g);
        let elite: Labels = vec![0, 0, 0, 1, 1, 1];
        update_weights_floor(&g, &mut w, &[&elite], 1.0, 0.0);
        for u in 0..g.n {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            for (&v, &wp) in g.adj[start..end].iter().zip(&w[start..end]) {
                let same = (u < 3) == ((v as usize) < 3);
                assert!((wp - if same { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn the_floor_stops_a_cut_edge_reaching_zero() {
        let g = two_triangles();
        let elite: Labels = vec![0, 0, 0, 1, 1, 1];
        for (floor, want_zero) in [(0.0, true), (0.05, false)] {
            let mut w = init_weights(&g);
            // Twenty rounds at the shipped rate is far past the point where an
            // unfloored cut edge is numerically dead.
            for _ in 0..20 {
                update_weights_floor(&g, &mut w, &[&elite], 0.5, floor);
            }
            let cut = w.iter().copied().fold(f64::INFINITY, f64::min);
            assert_eq!(
                cut < 1e-3,
                want_zero,
                "floor {floor} left the cut edge at {cut}"
            );
            assert!(cut >= floor - 1e-12, "floor {floor} violated: {cut}");
        }
    }
}
