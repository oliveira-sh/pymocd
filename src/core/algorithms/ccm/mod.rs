//! NSGA-III-CCM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021): a
//! self-contained reimplementation of the paper's method, not a thin wrapper
//! over this repo's optimized engine. Reproduced: the locus-based (Pizzuti
//! GA-Net style) genome with decode-by-union-find (`locus.rs`); CCM's own
//! single-threaded NSGA-III loop with Das–Dennis reference points and
//! niche-preserving environmental selection (`engine.rs`); both paper
//! customizations (the duplicate-permutation filter and the single-community
//! exclusion); adjacency-constrained crossover/mutation; the paper's exact
//! (Community Score, Community Fitness, Modularity) objectives; and the
//! max-modularity rank-1 decision rule. Removed, deliberately: the shared
//! NSGA-III engine (`core::metaheuristics::nsga3`, i.e. its `evolve` entry
//! point), the shared label-map operators, and data-parallel (Rayon-based)
//! evaluation — this baseline's cost and behaviour must track the paper, not
//! this repo's optimizations.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::objectives::community_score_fitness::community_objectives;
use crate::core::metaheuristics::helpers::operators::get_modularity_from_partition;
use crate::core::utils::normalize_community_ids;

use std::cmp::Ordering;

mod defaults;
mod engine;
mod locus;

pub use defaults::*;

/// (Community Score, Community Fitness, Modularity) are all maximized by the
/// paper; this engine's `Individual::dominates` assumes minimization, so the
/// objective vector fed to it is `(-CS, -CF, -Q)`.
fn evaluate(graph: &Graph, partition: &Partition, r: f64, alpha: f64) -> Vec<f64> {
    let (cs, cf) = community_objectives(graph, partition, r, alpha);
    let q = get_modularity_from_partition(partition, graph);
    vec![-cs, -cf, -q]
}

/// Run NSGA-III-CCM and return the **max-modularity** member of the rank-1
/// Pareto front (Shaik et al. recommend modularity-only decision-making when no
/// ground truth is available), normalized (isolated nodes → community `-1`).
#[allow(clippy::too_many_arguments)]
pub fn ccm(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> Partition {
    let nodes = graph.nodes_vec().clone();
    let index_of = locus::build_index(&nodes);

    let mut pop = engine::evolve(
        graph,
        &nodes,
        &index_of,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        divisions,
        |partition| evaluate(graph, partition, r, alpha),
    );

    engine::fast_non_dominated_sort(&mut pop);
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
        let res = ccm(&g, 100, 100, 0.8, 0.2, 1.0, 1.0, 12);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
