//! The sparse edge similarity: unit initialisation and the elite-consensus
//! reinforcement that closes the co-evolutionary loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;
use rayon::prelude::*;

use super::Labels;

pub fn init_weights(g: &CsrGraph) -> Vec<f64> {
    vec![1.0f64; g.adj.len()]
}

pub fn update_weights(g: &CsrGraph, wadj: &mut [f64], elites: &[&Labels], rho: f64) {
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
    wadj.par_iter_mut().zip(cov.par_iter()).for_each(|(w, &c)| {
        *w = (1.0 - rho) * *w + rho * c;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::fixtures::two_triangles;

    #[test]
    fn update_weights_raises_intra_lowers_inter() {
        let g = two_triangles();
        let mut w = init_weights(&g);
        let elite: Labels = vec![0, 0, 0, 1, 1, 1];
        update_weights(&g, &mut w, &[&elite], 1.0);
        for u in 0..g.n {
            let start = g.xadj[u] as usize;
            let end = g.xadj[u + 1] as usize;
            for (&v, &wp) in g.adj[start..end].iter().zip(&w[start..end]) {
                let same = (u < 3) == ((v as usize) < 3);
                assert!((wp - if same { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }
}
