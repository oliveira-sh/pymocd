//! operators/mutation.rs
//! Highly optimized mutation function for genetic algorithm
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rand::{distr::Bernoulli, prelude::Distribution};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

pub fn mutate(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    if mutation_rate == 0.0 || partition.is_empty() {
        return;
    }

    let mut rng = rand::rng();
    let mutation_dist = Bernoulli::new(mutation_rate).unwrap();
    let nodes_to_mutate: Vec<NodeId> = if mutation_rate > 0.5 {
        // If high mutation rate, it's faster to collect all and then filter
        partition
            .keys()
            .copied()
            .filter(|_| mutation_dist.sample(&mut rng))
            .collect()
    } else {
        // For low mutation rates, early filtering is more efficient
        let mut nodes = Vec::with_capacity((partition.len() as f64 * mutation_rate * 1.2) as usize);
        for &node in partition.keys() {
            if mutation_dist.sample(&mut rng) {
                nodes.push(node);
            }
        }
        nodes
    };

    if nodes_to_mutate.is_empty() {
        return;
    }

    if nodes_to_mutate.len() > 128 {
        parallel_mutate(partition, graph, &nodes_to_mutate);
    } else {
        sequential_mutate(partition, graph, &nodes_to_mutate);
    }
}

fn sequential_mutate(partition: &mut Partition, graph: &Graph, nodes_to_mutate: &[NodeId]) {
    let mut community_freq = FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

    for &node in nodes_to_mutate {
        community_freq.clear();

        if let Some(neighbors) = graph.adjacency_list.get(&node) {
            let mut max_count = 0;
            let mut best_community = partition[&node]; // Current community as fallback

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
        .par_chunks(64)
        .flat_map(|chunk| {
            let mut local_updates = Vec::with_capacity(chunk.len());
            let mut community_freq = FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

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
