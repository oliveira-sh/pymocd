//! operators/population.rs
//! Make the initial population  in the Genetic Algorithm
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};

use rand::Rng;
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeMap;

#[allow(dead_code)]
pub fn generate_optimized_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    let mut rng = rand::rng();
    let nodes: Vec<NodeId> = graph.nodes.iter().copied().collect();
    let num_nodes = nodes.len();
    let mut population = Vec::with_capacity(population_size);

    // 1. Random Partitions (33% of population)
    for _ in 0..population_size / 3 {
        let mut partition = BTreeMap::new();
        for &node in &nodes {
            partition.insert(node, rng.random_range(0..=(num_nodes / 2)) as CommunityId);
        }
        population.push(partition);
    }

    // 2. Neighbor-based Partitions (33% of population)
    for _ in 0..population_size / 3 {
        let mut partition = BTreeMap::new();
        let mut unassigned: HashSet<_> = nodes.iter().copied().collect();
        let mut current_community: CommunityId = 0;

        while !unassigned.is_empty() {
            let start_node = *unassigned.iter().next().unwrap();
            let mut to_process = vec![start_node];

            while let Some(node) = to_process.pop() {
                if unassigned.remove(&node) {
                    partition.insert(node, current_community);

                    // Add some neighbors with 70% probability
                    for &neighbor in graph.neighbors(&node) {
                        if unassigned.contains(&neighbor) && rng.random_bool(0.7) {
                            to_process.push(neighbor);
                        }
                    }
                }
            }
            current_community += 1;
        }
        population.push(partition);
    }

    // 3. Single-community and Small-community Partitions (remaining population)
    for i in 2 * population_size / 3..population_size {
        let mut partition = BTreeMap::new();
        if i % 2 == 0 {
            // Single community
            for &node in &nodes {
                partition.insert(node, 0);
            }
        } else {
            // Two communities based on degree
            for &node in &nodes {
                let degree = graph.neighbors(&node).len();
                partition.insert(node, if degree > nodes.len() / 2 { 0 } else { 1 });
            }
        }
        population.push(partition);
    }

    population
}

#[allow(dead_code)]
pub fn generate_initial_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    let mut rng = rand::rng();
    let nodes: Vec<NodeId> = graph.nodes.iter().copied().collect();
    let num_nodes = nodes.len();

    (0..population_size)
        .map(|_| {
            nodes
                .iter()
                .map(|&node| (node, rng.random_range(0..num_nodes) as CommunityId))
                .collect()
        })
        .collect()
}
