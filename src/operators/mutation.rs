//! operators/mutation.rs
//! Genetic Algorithm population mutation function
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};

use rand::Rng;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;

pub fn optimized_mutate(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    let mut rng = rand::rng();

    // Convert BTreeMap to a faster hash map for the duration of the mutation
    let partition_size = partition.len();
    let mut fast_partition: HashMap<NodeId, CommunityId> =
        HashMap::with_capacity_and_hasher(partition_size, Default::default());
    fast_partition.extend(partition.iter().map(|(&k, &v)| (k, v)));

    // Pre-calculate nodes to mutate
    let nodes: Vec<NodeId> = fast_partition
        .keys()
        .copied()
        .filter(|_| rng.random_bool(mutation_rate))
        .collect();

    // Pre-allocate community cache with expected size
    let mut community_cache: HashMap<NodeId, HashMap<CommunityId, usize>> =
        HashMap::with_capacity_and_hasher(nodes.len(), FxBuildHasher);

    // Process nodes in batches for better cache locality
    const BATCH_SIZE: usize = 64;
    for node_chunk in nodes.chunks(BATCH_SIZE) {
        for &node in node_chunk {
            let neighbor_communities = community_cache.entry(node).or_insert_with(|| {
                let mut freq = HashMap::default();
                if let Some(neighbors) = graph.adjacency_list.get(&node) {
                    freq.reserve(neighbors.len());
                    for &neighbor in neighbors {
                        if let Some(&community) = fast_partition.get(&neighbor) {
                            *freq.entry(community).or_insert(0) += 1;
                        }
                    }
                }
                freq
            });

            if let Some((&new_community, _)) =
                neighbor_communities.iter().max_by_key(|&(_, count)| count)
            {
                fast_partition.insert(node, new_community);
            }
        }
    }
    // Update
    partition.clear();
    partition.extend(fast_partition);
}

pub fn polynomial_mutation(
    partition: &mut Partition,
    graph: &Graph,
    mutation_rate: f64,
    eta_m: f64,
) {
    let mut rng = rand::rng();
    let nodes: Vec<NodeId> = graph.nodes.iter().cloned().collect();

    let mut fast_partition: HashMap<NodeId, CommunityId> = partition.iter().map(|(&k, &v)| (k, v)).collect();

    for &node in &nodes {
        if rng.random::<f64>() > mutation_rate {
            continue;
        }

        let neighbor_communities: HashMap<CommunityId, usize> = match graph.adjacency_list.get(&node) {
            Some(neighbors) => {
                let mut freq = HashMap::default();
                for &neighbor in neighbors {
                    if let Some(&comm) = fast_partition.get(&neighbor) {
                        *freq.entry(comm).or_insert(0) += 1;
                    }
                }
                freq
            }
            None => continue,
        };

        if neighbor_communities.is_empty() {
            continue;
        }

        let adjusted: Vec<(CommunityId, f64)> = neighbor_communities.iter()
            .map(|(&comm, &count)| (comm, (count as f64).powf(1.0 / (eta_m + 1.0))))
            .collect();

        let total: f64 = adjusted.iter().map(|(_, val)| val).sum();

        if total <= 0.0 {
            continue;
        }

        let mut r = rng.random::<f64>() * total;
        let mut selected_comm = None;
        for &(comm, val) in &adjusted {
            if r <= val {
                selected_comm = Some(comm);
                break;
            }
            r -= val;
        }

        if let Some(comm) = selected_comm {
            fast_partition.insert(node, comm);
        }
    }

    partition.clear();
    partition.extend(fast_partition);
}