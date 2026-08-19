//! Mutation: move each drawn node to the community most of its neighbours hold.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::core::algorithms::mopots::sampling::bernoulli;
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};

const PARALLEL_NODE_THRESHOLD: usize = 128;
const PARALLEL_CHUNK: usize = 64;
const DRAW_CAPACITY_SLACK: f64 = 1.2;

type CommunityCounts = FxHashMap<CommunityId, usize>;

pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64, rng: &mut StdRng) {
    if mutation_rate == 0.0 || partition.is_empty() {
        return;
    }

    let nodes = graph.nodes_vec();
    let mutation_dist = bernoulli(mutation_rate);

    let expected = (nodes.len() as f64 * mutation_rate * DRAW_CAPACITY_SLACK) as usize;
    let mut nodes_to_mutate: Vec<NodeId> = Vec::with_capacity(expected.min(nodes.len()));
    for &node in nodes {
        if rng.sample(mutation_dist) {
            nodes_to_mutate.push(node);
        }
    }

    if nodes_to_mutate.is_empty() {
        return;
    }

    // the two paths compute different things and the input alone picks one, never the thread count
    if nodes_to_mutate.len() > PARALLEL_NODE_THRESHOLD {
        parallel_mutate(partition, graph, &nodes_to_mutate);
    } else {
        sequential_mutate(partition, graph, &nodes_to_mutate);
    }
}

// Gauss-Seidel: a node already sees the moves made earlier in this same sweep
fn sequential_mutate(partition: &mut Partition, graph: &Graph, nodes_to_mutate: &[NodeId]) {
    let mut counts = new_counts();

    for &node in nodes_to_mutate {
        if let Some(community) = majority_move(partition, graph, node, &mut counts) {
            partition.insert(node, community);
        }
    }
}

// Jacobi: every move is computed against the pre-sweep labels and applied afterwards
fn parallel_mutate(partition: &mut Partition, graph: &Graph, nodes_to_mutate: &[NodeId]) {
    let updates: Vec<(NodeId, CommunityId)> = nodes_to_mutate
        .par_chunks(PARALLEL_CHUNK)
        .flat_map(|chunk| {
            let mut local_updates = Vec::with_capacity(chunk.len());
            let mut counts = new_counts();

            for &node in chunk {
                if let Some(community) = majority_move(partition, graph, node, &mut counts) {
                    local_updates.push((node, community));
                }
            }

            local_updates
        })
        .collect();

    for (node, community) in updates {
        partition.insert(node, community);
    }
}

/// The community most of `node`'s neighbours hold, or `None` when it already
/// holds it, when no neighbour is labelled, or when the node is not in `partition`.
fn majority_move(
    partition: &Partition,
    graph: &Graph,
    node: NodeId,
    counts: &mut CommunityCounts,
) -> Option<CommunityId> {
    counts.clear();

    let &current_community = partition.get(&node)?;
    let neighbors = graph.adjacency_list.get(&node)?;

    let mut max_count = 0;
    let mut best_community = current_community;

    // finalize() sorts the neighbour list, so the first community to peak always wins
    for &neighbor in neighbors {
        if let Some(&community) = partition.get(&neighbor) {
            let count = counts.entry(community).or_insert(0);
            *count += 1;

            if *count > max_count {
                max_count = *count;
                best_community = community;
            }
        }
    }

    (max_count > 0 && best_community != current_community).then_some(best_community)
}

fn new_counts() -> CommunityCounts {
    FxHashMap::with_capacity_and_hasher(16, FxBuildHasher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopots::fixtures::two_triangles;
    use crate::core::algorithms::mopots::operators::generate_population;
    use crate::core::algorithms::mopots::sampling::slot_rng;

    #[test]
    fn mutation_is_reproducible() {
        let graph = two_triangles();
        let base = generate_population(&graph, 1).remove(0);

        let mut a = base.clone();
        let mut b = base;
        mutation(&mut a, &graph, 0.5, &mut slot_rng(7, 0));
        mutation(&mut b, &graph, 0.5, &mut slot_rng(7, 0));
        assert_eq!(a, b);
    }
}
