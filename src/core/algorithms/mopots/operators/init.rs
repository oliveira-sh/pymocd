//! The initial population: one uniformly random fine labelling per slot.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;

use crate::core::algorithms::mopots::sampling::slot_rng;
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};

// must not collide with any generation index: create_offspring salts its streams with those
const INIT_SALT: u64 = 0x5EED_0001;

pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    // finalize() sorts node_vec, so the label order is the same on every run
    let node_ids = graph.nodes_vec();

    if node_ids.is_empty() {
        return (0..population_size).map(|_| Partition::default()).collect();
    }

    let num_communities = node_ids.len();
    (0..population_size)
        .map(|slot| {
            let mut rng = slot_rng(INIT_SALT, slot);
            random_partition(node_ids, num_communities, &mut rng)
        })
        .collect()
}

fn random_partition(node_ids: &[NodeId], num_communities: usize, rng: &mut StdRng) -> Partition {
    node_ids
        .iter()
        .map(|&node_id| {
            let community = rng.random_range(0..num_communities) as CommunityId;
            (node_id, community)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopots::fixtures::two_triangles;

    #[test]
    fn generate_population_is_reproducible() {
        let graph = two_triangles();
        let a = generate_population(&graph, 8);
        let b = generate_population(&graph, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn generate_population_slot_is_independent() {
        let graph = two_triangles();
        let full = generate_population(&graph, 8);

        let alone = generate_population(&graph, 1);
        assert_eq!(alone.len(), 1);
        assert_eq!(alone[0], full[0]);

        for (slot, partition) in full.iter().enumerate() {
            let mut rng = slot_rng(INIT_SALT, slot);
            let expected = random_partition(graph.nodes_vec(), graph.num_nodes(), &mut rng);
            assert_eq!(*partition, expected);
        }
    }

    #[test]
    fn generate_population_labels_every_node() {
        let graph = two_triangles();
        for partition in generate_population(&graph, 4) {
            assert_eq!(partition.len(), graph.num_nodes());
            for &node in graph.nodes_vec() {
                assert!(partition.contains_key(&node));
            }
        }
    }

    #[test]
    fn generate_population_on_empty_graph_is_empty() {
        let graph = Graph::new();
        let population = generate_population(&graph, 3);
        assert_eq!(population.len(), 3);
        assert!(population.iter().all(Partition::is_empty));
    }
}
