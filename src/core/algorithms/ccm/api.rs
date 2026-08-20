//! Public entry points for NSGA-III-CCM.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::cmp::Ordering;

use crate::core::graph::{Graph, NodeId, Partition, normalize_community_ids};

use super::locus;
use super::nsga3;
use super::objectives::{evaluate, modularity_labels};

/// The **max-modularity** member of the rank-1 Pareto front — Shaik et al.'s
/// decision rule when no ground truth is available. Isolated nodes come back as
/// community `-1`.
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
    let (nodes, neighbor_pos, front) = rank_one_front(
        graph, pop_size, num_gens, cross_rate, mut_rate, r, alpha, divisions,
    );

    let best = front
        .iter()
        .map(|labels| (modularity_labels(&neighbor_pos, labels), labels))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    to_partition(graph, &nodes, best)
}

/// The whole rank-1 front `ccm` selects from: the paper's Table 1/2 protocol
/// reports the best-NMI *and* the best-Q member, so it needs the candidate set.
#[allow(clippy::too_many_arguments)]
pub fn ccm_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> Vec<Partition> {
    let (nodes, _, front) = rank_one_front(
        graph, pop_size, num_gens, cross_rate, mut_rate, r, alpha, divisions,
    );
    front
        .iter()
        .map(|labels| to_partition(graph, &nodes, labels))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn rank_one_front(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> (Vec<NodeId>, Vec<Vec<usize>>, Vec<Vec<i32>>) {
    let nodes = graph.nodes_vec().clone();
    let neighbor_pos = locus::neighbor_positions(graph, &nodes);

    let mut pop = nsga3::evolve(
        &neighbor_pos,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        divisions,
        |labels| evaluate(&neighbor_pos, labels, r, alpha),
    );
    nsga3::fast_non_dominated_sort(&mut pop);

    let front = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| ind.labels.clone())
        .collect();
    (nodes, neighbor_pos, front)
}

fn to_partition(graph: &Graph, nodes: &[NodeId], labels: &[i32]) -> Partition {
    let partition: Partition = nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, labels[p]))
        .collect();
    normalize_community_ids(graph, partition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ccm::fixtures::two_triangles;

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
