//! MOGA-Net (Pizzuti, IEEE ICTAI 2009 / IEEE TEC 16(3):418-430, 2012): a
//! self-contained, single-threaded reimplementation of the paper's mechanism —
//! locus-based adjacency genome (Park & Song 1989) with safe initialization
//! and repaired operators (`locus.rs`, `operators.rs`); the MATLAB
//! `gamultiobj`-style elitist + roulette generational replacement the paper
//! used, *not* combine-parents-and-offspring-then-truncate (`engine.rs`);
//! (Community Score, Community Fitness) bi-objective; max-modularity rank-1
//! decision rule. Deliberately avoids the shared NSGA-II engine and Rayon so
//! this baseline's cost tracks the published method.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::normalize_community_ids;
use crate::core::graph::{Graph, Partition};
use std::cmp::Ordering;

mod defaults;
mod engine;
mod individual;
mod locus;
mod objectives;
mod operators;

pub use defaults::*;

use individual::fast_non_dominated_sort;
use locus::Locus;

/// Newman modularity `Q = Σ_c l_c/m − (d_c/2m)²` on a position-indexed label
/// array (matches `core::metrics::modularity`; Vec accumulators, ascending
/// position / community-id order for deterministic summation).
fn label_modularity(graph: &Graph, locus: &Locus, labels: &[i32]) -> f64 {
    let m = graph.num_edges() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let n_comms = labels.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut l_c = vec![0.0f64; n_comms]; // internal edges, counted once
    let mut d_c = vec![0.0f64; n_comms]; // total degree
    for (p, &lab) in labels.iter().enumerate() {
        let c = lab as usize;
        d_c[c] += locus.neighbors[p].len() as f64;
        for &q in &locus.neighbors[p] {
            if p < q && labels[q] == lab {
                l_c[c] += 1.0;
            }
        }
    }
    (0..n_comms)
        .map(|c| l_c[c] / m - (d_c[c] / (2.0 * m)).powi(2))
        .sum()
}

/// Run MOGA-Net and return the **max-modularity** member of the rank-1 Pareto
/// front (Pizzuti 2012, Sec. V-E), normalized (isolated nodes → community `-1`).
pub fn moga_net(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> Partition {
    let locus = Locus::build(graph);
    let mut pop = engine::run(&locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha);

    fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (label_modularity(graph, &locus, &ind.labels), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    // Shared Partition type appears only here, at the module boundary.
    let partition: Partition = locus
        .nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, best.labels[p]))
        .collect();
    normalize_community_ids(graph, partition)
}

/// The rank-1 Pareto front `moga_net` selects from, as normalized partitions —
/// Pizzuti's Table 1 protocol (best-NMI over the front) needs it.
pub fn moga_net_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> Vec<Partition> {
    let locus = Locus::build(graph);
    let mut pop = engine::run(&locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha);

    fast_non_dominated_sort(&mut pop);
    pop.iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| {
            let partition: Partition = locus
                .nodes
                .iter()
                .enumerate()
                .map(|(p, &node)| (node, ind.labels[p]))
                .collect();
            normalize_community_ids(graph, partition)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::objectives::community_objectives;
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn cs_maximal_for_two_community_split() {
        let g = two_triangles();
        // nodes_vec is sorted, so position p == node id p here.
        let locus = Locus::build(&g);
        let split = vec![0, 0, 0, 1, 1, 1];
        let one = vec![0, 0, 0, 0, 0, 0];
        let singletons = vec![0, 1, 2, 3, 4, 5];
        let (cs_split, cf_split) = community_objectives(&locus, &split, 2.0, 1.0);
        let (cs_one, cf_one) = community_objectives(&locus, &one, 2.0, 1.0);
        let (cs_sing, cf_sing) = community_objectives(&locus, &singletons, 2.0, 1.0);
        assert!(cs_split > cs_one, "CS split {cs_split} !> one {cs_one}");
        assert!(
            cs_split > cs_sing,
            "CS split {cs_split} !> singletons {cs_sing}"
        );
        // v_S double-counts edges: each triangle has 3 internal edges → v_S=6,
        // M(S)=(2/3)^2, score=6·4/9, CS=2·(8/3).
        assert!((cs_split - 16.0 / 3.0).abs() < 1e-9, "CS={cs_split}");
        // CF is MAXIMIZED (internal cohesion): singletons → CF=0, the coarse
        // 'one' blob scores highest, so CS and CF genuinely trade off.
        assert_eq!(cf_sing, 0.0);
        assert!(cf_one > cf_split, "coarser partition must win the CF axis");
    }

    #[test]
    fn finds_two_community_split() {
        let g = two_triangles();
        let res = moga_net(&g, 100, 80, 0.8, 0.2, 2.0, 1.0);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
