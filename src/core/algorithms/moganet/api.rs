//! Public entry points for MOGA-Net.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::cmp::Ordering;

use crate::core::graph::{Graph, Partition, normalize_community_ids};

use super::locus::Locus;
use super::objectives::label_modularity;
use super::search::{Individual, fast_non_dominated_sort, run};

fn evolve_ranked(
    locus: &Locus,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> Vec<Individual> {
    let mut pop = run(locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha);
    fast_non_dominated_sort(&mut pop);
    pop
}

fn to_partition(graph: &Graph, locus: &Locus, labels: &[i32]) -> Partition {
    let partition: Partition = locus
        .nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, labels[p]))
        .collect();
    normalize_community_ids(graph, partition)
}

/// The max-modularity member of the rank-1 Pareto front (Pizzuti 2012,
/// Sec. V-E), normalized (isolated nodes -> community `-1`).
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
    let pop = evolve_ranked(&locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha);

    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (label_modularity(graph, &locus, &ind.labels), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    to_partition(graph, &locus, &best.labels)
}

/// The whole rank-1 front `moga_net` selects from, as normalized partitions;
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
    let pop = evolve_ranked(&locus, pop_size, num_gens, cross_rate, mut_rate, r, alpha);

    pop.iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| to_partition(graph, &locus, &ind.labels))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::moganet::objectives::community_objectives;

    fn two_triangles_joined_by_one_bridge() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn cs_maximal_for_two_community_split() {
        let g = two_triangles_joined_by_one_bridge();
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
        // v_S double-counts edges: each triangle has 3 internal edges -> v_S=6,
        // M(S)=(2/3)^2, score=6*4/9, CS=2*(8/3).
        assert!((cs_split - 16.0 / 3.0).abs() < 1e-9, "CS={cs_split}");
        // CF is MAXIMIZED, so singletons score 0 and the coarse blob wins that
        // axis: the two objectives genuinely trade off.
        assert_eq!(cf_sing, 0.0);
        assert!(cf_one > cf_split, "coarser partition must win the CF axis");
    }

    #[test]
    fn finds_two_community_split() {
        let g = two_triangles_joined_by_one_bridge();
        let res = moga_net(&g, 100, 80, 0.8, 0.2, 2.0, 1.0);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
