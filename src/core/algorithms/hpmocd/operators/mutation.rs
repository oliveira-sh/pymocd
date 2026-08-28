//! Neighbour-majority mutation: a drawn node takes the community most of its neighbours hold.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{distr::Bernoulli, prelude::*};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};

const DRAW_CAPACITY_SLACK: f64 = 1.2;
const PRESIZED_DRAW_MAX_RATE: f64 = 0.5;
const PARALLEL_NODE_THRESHOLD: usize = 128;
const PARALLEL_CHUNK: usize = 64;
const FREQ_CAPACITY: usize = 16;

/// One Bernoulli(`mutation_rate`) draw per node in `partition.keys()` order; the
/// branches differ only in how the result is allocated.
fn draw_nodes_to_mutate(partition: &Partition, mutation_rate: f64) -> Vec<NodeId> {
    let mut rng = rand::rng();
    let mutation_dist = Bernoulli::new(mutation_rate).unwrap();

    if mutation_rate > PRESIZED_DRAW_MAX_RATE {
        partition
            .keys()
            .copied()
            .filter(|_| mutation_dist.sample(&mut rng))
            .collect()
    } else {
        let expected = partition.len() as f64 * mutation_rate * DRAW_CAPACITY_SLACK;
        let mut nodes = Vec::with_capacity(expected as usize);
        for &node in partition.keys() {
            if mutation_dist.sample(&mut rng) {
                nodes.push(node);
            }
        }
        nodes
    }
}

pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    if mutation_rate == 0.0 || partition.is_empty() {
        return;
    }

    let nodes_to_mutate = draw_nodes_to_mutate(partition, mutation_rate);
    if nodes_to_mutate.is_empty() {
        return;
    }

    // two different operators, not two speeds of one: the sequential sweep sees moves
    // made earlier in it, the parallel one scores every node against the pre-sweep labels
    if nodes_to_mutate.len() > PARALLEL_NODE_THRESHOLD {
        parallel_mutate(partition, graph, &nodes_to_mutate);
    } else {
        sequential_mutate(partition, graph, &nodes_to_mutate);
    }
}

fn sequential_mutate(partition: &mut Partition, graph: &Graph, nodes_to_mutate: &[NodeId]) {
    let mut community_freq = FxHashMap::with_capacity_and_hasher(FREQ_CAPACITY, FxBuildHasher);

    for &node in nodes_to_mutate {
        community_freq.clear();

        if let Some(neighbors) = graph.adjacency_list.get(&node) {
            let mut max_count = 0;
            let mut best_community = partition[&node];

            for &neighbor in neighbors {
                if let Some(&community) = partition.get(&neighbor) {
                    let count = community_freq.entry(community).or_insert(0);
                    *count += 1;

                    if *count > max_count {
                        max_count = *count;
                        best_community = community;
                    }
                }
            }

            if max_count > 0 && best_community != partition[&node] {
                partition.insert(node, best_community);
            }
        }
    }
}

fn parallel_mutate(partition: &mut Partition, graph: &Graph, nodes_to_mutate: &[NodeId]) {
    let updates: Vec<(NodeId, CommunityId)> = nodes_to_mutate
        .par_chunks(PARALLEL_CHUNK)
        .flat_map(|chunk| {
            let mut local_updates = Vec::with_capacity(chunk.len());
            let mut community_freq =
                FxHashMap::with_capacity_and_hasher(FREQ_CAPACITY, FxBuildHasher);

            for &node in chunk {
                community_freq.clear();

                if let Some(neighbors) = graph.adjacency_list.get(&node) {
                    let current_community = partition[&node];
                    let mut max_count = 0;
                    let mut best_community = current_community;

                    for &neighbor in neighbors {
                        if let Some(&community) = partition.get(&neighbor) {
                            let count = community_freq.entry(community).or_insert(0);
                            *count += 1;

                            if *count > max_count {
                                max_count = *count;
                                best_community = community;
                            }
                        }
                    }

                    if max_count > 0 && best_community != current_community {
                        local_updates.push((node, best_community));
                    }
                }
            }

            local_updates
        })
        .collect();
    for (node, community) in updates {
        partition.insert(node, community);
    }
}
