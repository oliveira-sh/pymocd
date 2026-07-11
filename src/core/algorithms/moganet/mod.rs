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

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::objectives::decomposed_modularity::calculate_objectives;
use crate::core::graph::normalize_community_ids;
use std::cmp::Ordering;

mod defaults;
mod engine;
mod individual;
mod locus;
mod operators;

pub use defaults::*;

use individual::fast_non_dominated_sort;
use locus::Locus;

/// Q = 1 - intra - inter (Shi et al. 2012 decomposed modularity, mathematically
/// the standard Newman-Girvan modularity); single-threaded (`parallel = false`).
fn modularity(graph: &Graph, partition: &Partition) -> f64 {
    let m = calculate_objectives(graph, partition, graph.precompute_degrees(), false);
    1.0 - m.intra - m.inter
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
    let mut pop = engine::run(
        graph, &locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha,
    );

    fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (modularity(graph, &ind.partition), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    normalize_community_ids(graph, best.partition.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{CommunityId, NodeId};
    use crate::core::metaheuristics::helpers::objectives::community_score_fitness::community_objectives;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    fn part(pairs: &[(NodeId, CommunityId)]) -> Partition {
        pairs.iter().copied().collect()
    }

    #[test]
    fn cs_maximal_for_two_community_split() {
        let g = two_triangles();
        let split = part(&[(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]);
        let one = part(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
        let singletons = part(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
        let (cs_split, cf_split) = community_objectives(&g, &split, 2.0, 1.0);
        let (cs_one, cf_one) = community_objectives(&g, &one, 2.0, 1.0);
        let (cs_sing, cf_sing) = community_objectives(&g, &singletons, 2.0, 1.0);
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
