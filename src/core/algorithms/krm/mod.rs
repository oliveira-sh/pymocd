//! NSGA-III-KRM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021): a
//! self-contained, faithful reimplementation of the paper's method, not a
//! wrapper around this repo's optimized engine. Reproduced: the locus-based
//! ("GA-Net"/Pizzuti) genome and union-find decode (`locus.rs`); a
//! from-scratch, single-threaded NSGA-III loop (Deb & Jain 2014) with
//! Das-Dennis reference points and niche-preserving environmental selection
//! (`nsga3.rs`); the paper's two customizations -- the duplicate-permutation
//! filter and the single-community exclusion; adjacency-constrained
//! crossover/mutation (`operators.rs`); the paper's exact (KKM, RC, Q)
//! objective formulas; and the max-modularity rank-1 decision rule. Removed:
//! the shared `core::metaheuristics::nsga3` module's entry point, the shared
//! label-map crossover/mutation operators, and data-parallel (Rayon-based)
//! evaluation -- this detector now runs single-threaded so its cost reflects
//! the published method, not this repo's optimizations.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::operators::get_modularity_from_partition;
use crate::core::utils::normalize_community_ids;
use std::cmp::Ordering;

mod defaults;
mod individual;
mod locus;
mod nsga3;
mod operators;

pub use defaults::*;

use individual::fast_non_dominated_sort;
use locus::Locus;

/// Run NSGA-III-KRM and return the **max-modularity** member of the rank-1
/// Pareto front (Shaik et al. recommend modularity-only decision-making when
/// no ground truth is available), normalized (isolated nodes → community `-1`).
pub fn krm(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Partition {
    let locus = Locus::build(graph);
    let mut pop = nsga3::run(graph, &locus, pop_size, num_gens, cross_rate, mut_rate, divisions);

    fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (get_modularity_from_partition(&ind.partition, graph), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    normalize_community_ids(graph, best.partition.clone())
}

#[cfg(test)]
mod tests {
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
    fn finds_two_community_split() {
        let g = two_triangles();
        let res = krm(&g, 100, 100, 0.8, 0.2, 12);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
