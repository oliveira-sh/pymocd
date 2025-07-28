//! leoben_algorithm/population.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rand::{Rng, rng};

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

    let node_ids = graph.nodes_vec(); // Vec<NodeId>
    let num_communities = node_ids.len();

    (0..population_size)
        .map(|_| random_partition(&node_ids, num_communities, &mut rng))
        .collect()
}
