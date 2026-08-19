//! Consensus crossover: every node takes the label most of the parents give it.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::core::graph::{CommunityId, NodeId, Partition};

/// Per node, the label most of `parents` give it; a tie is drawn uniformly from
/// the tied labels. `nodes` must be sorted: it fixes the order of those draws.
pub fn ensemble_crossover(parents: &[&Partition], nodes: &[NodeId], rng: &mut StdRng) -> Partition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let mut child = FxHashMap::with_capacity_and_hasher(nodes.len(), FxBuildHasher);

    let mut community_counts = FxHashMap::with_capacity_and_hasher(parents.len(), FxBuildHasher);
    let mut candidates = Vec::with_capacity(parents.len());

    for &node in nodes {
        community_counts.clear();

        let majority_threshold = parents.len() / 2 + 1;
        let mut max_count = 0;
        // a node no parent labels keeps its own id, which is the singleton label
        let mut best_community = parents[0]
            .get(&node)
            .copied()
            .unwrap_or(node as CommunityId);

        for parent in parents {
            if let Some(&community) = parent.get(&node) {
                let count = community_counts.entry(community).or_insert(0);
                *count += 1;

                // an unbeatable majority stops the count, so real ties still reach the tie logic
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
            // sorting drops the hash order, so the drawn index means the same thing every run
            candidates.sort_unstable();

            best_community = candidates[rng.random_range(0..candidates.len())];
        }

        child.insert(node, best_community);
    }

    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopots::fixtures::two_triangles;
    use crate::core::algorithms::mopots::operators::generate_population;
    use crate::core::algorithms::mopots::sampling::slot_rng;

    #[test]
    fn ensemble_crossover_is_reproducible() {
        let graph = two_triangles();
        let population = generate_population(&graph, 4);
        let parents: Vec<&Partition> = population.iter().collect();
        let nodes = graph.nodes_vec();

        let a = ensemble_crossover(&parents, nodes, &mut slot_rng(9, 0));
        let b = ensemble_crossover(&parents, nodes, &mut slot_rng(9, 0));
        assert_eq!(a, b);
        assert_eq!(a.len(), nodes.len());
    }
}
