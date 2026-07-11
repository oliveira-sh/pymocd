//! Newman modularity Q via Shi's decomposition (Q = 1 − intra − inter).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use rustc_hash::FxHashMap;

/// Modularity Q of `partition` on `graph`:
/// `Q = Σ_c l_c/m − (d_c/2m)²` (each community's internal edges counted once).
pub fn modularity(graph: &Graph, partition: &Partition) -> f64 {
    let m = graph.edges.len() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let mut communities: FxHashMap<i32, Vec<NodeId>> = FxHashMap::default();
    for (&node, &comm) in partition.iter() {
        communities.entry(comm).or_default().push(node);
    }

    let mut intra_sum = 0.0f64;
    let mut inter = 0.0f64;
    for nodes in communities.values() {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;
        for &node in nodes {
            community_degree += *graph.degrees.get(&node).unwrap_or(&0) as f64;
            if let Some(neighbors) = graph.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    if node < neighbor && partition.get(&neighbor) == partition.get(&node) {
                        community_edges += 1.0;
                    }
                }
            }
        }
        intra_sum += community_edges;
        inter += (community_degree / (2.0 * m)).powi(2);
    }

    let intra = 1.0 - (intra_sum / m);
    1.0 - intra - inter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::karate::{KARATE_CLUB, karate_club};

    #[test]
    fn karate_club_matches_networkx() {
        let g = karate_club();
        let part: Partition = KARATE_CLUB
            .iter()
            .enumerate()
            .map(|(n, &c)| (n as i32, c))
            .collect();
        // networkx.community.modularity(G, club split, weight=None)
        assert!((modularity(&g, &part) - 0.358234714003945).abs() < 1e-12);
    }
}
