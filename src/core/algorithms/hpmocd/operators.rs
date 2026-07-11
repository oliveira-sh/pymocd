//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
//! Genetic operators for the label-map encoding: local-move mutation,
//! per-node majority-vote (consensus) crossover, and random initial population.

use super::objectives::{Metrics, calculate_objectives};
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rand::{Rng, RngExt, distr::Bernoulli, prelude::*, rng, rngs::ThreadRng, seq::IndexedRandom};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::HashMap;

pub fn get_fitness(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
    parallel: bool,
) -> Metrics {
    calculate_objectives(graph, partition, degrees, parallel)
}

pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    mutate(partition, graph, mutation_rate);
}

pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    generate_initial_population(graph, population_size)
}

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

pub fn ensemble_crossover(parents: &[&Partition], rng: &mut ThreadRng) -> Partition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let keys: Vec<NodeId> = parents[0].keys().copied().collect();
    let mut child = FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);

    let mut community_counts = FxHashMap::with_capacity_and_hasher(parents.len(), FxBuildHasher);
    let mut candidates = Vec::with_capacity(parents.len());

    for &node in &keys {
        community_counts.clear();

        let majority_threshold = parents.len().div_ceil(2);
        let mut max_count = 0;
        let mut best_community = parents[0][&node];

        for parent in parents {
            if let Some(&community) = parent.get(&node) {
                let count = community_counts.entry(community).or_insert(0);
                *count += 1;

                if *count > max_count {
                    max_count = *count;
                    best_community = community;
                    if *count >= majority_threshold {
                        break;
                    }
                }
            }
        }

        let tie_count = community_counts
            .values()
            .filter(|&&count| count == max_count)
            .count();

        if tie_count > 1 {
            candidates.clear();
            candidates.extend(
                community_counts
                    .iter()
                    .filter(|(_, count)| **count == max_count)
                    .map(|(&comm, _)| comm),
            );

            best_community = *candidates.choose(rng).unwrap();
        }

        child.insert(node, best_community);
    }

    child
}

fn random_partition(node_ids: &[NodeId], num_communities: usize, rng: &mut impl Rng) -> Partition {
    node_ids
        .iter()
        .map(|&node_id| {
            let community = rng.random_range(0..num_communities) as CommunityId;
            (node_id, community)
        })
        .collect()
}

pub fn generate_initial_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    let mut rng = rng();

    let node_ids: Vec<NodeId> = graph.nodes.iter().copied().collect();
    let num_communities = node_ids.len();
    (0..population_size)
        .map(|_| random_partition(&node_ids, num_communities, &mut rng))
        .collect()
}
