//! Public entry points: the selected partition, and the front it is chosen from.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::cmp::Ordering;

use crate::core::graph::{Graph, Partition, normalize_community_ids};

use super::locus::Locus;
use super::nsga3;

const NEGATED_MODULARITY: usize = 2;

/// The max-modularity member of the rank-1 front; isolated nodes come back as `-1`.
pub fn krm(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Partition {
    let locus = Locus::build(graph);
    let mut pop = nsga3::run(
        graph, &locus, pop_size, num_gens, cross_rate, mut_rate, divisions,
    );

    nsga3::fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (-ind.objectives[NEGATED_MODULARITY], ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    to_partition(graph, &locus, &best.labels)
}

/// The whole rank-1 front `krm` selects from.
pub fn krm_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Vec<Partition> {
    let locus = Locus::build(graph);
    let mut pop = nsga3::run(
        graph, &locus, pop_size, num_gens, cross_rate, mut_rate, divisions,
    );

    nsga3::fast_non_dominated_sort(&mut pop);
    pop.iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| to_partition(graph, &locus, &ind.labels))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::krm::fixtures::two_triangles;

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
