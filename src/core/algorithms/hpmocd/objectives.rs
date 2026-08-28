//! Shi's decomposed-modularity objectives and their value type.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::iter::*;
use rustc_hash::FxHashMap as HashMap;

use crate::core::graph::{Graph, NodeId, Partition};

const PARALLEL_COMMUNITY_THRESHOLD: usize = 8;

#[derive(Debug, Default, PartialEq)]
pub struct Metrics {
    pub intra: f64,
    pub inter: f64,
}

fn degree_sum(nodes: &[NodeId], degrees: &HashMap<NodeId, usize>) -> f64 {
    let mut community_degree = 0.0;
    for &node in nodes {
        let degree = *degrees.get(&node).unwrap_or(&0) as f64;
        community_degree += degree;
    }
    community_degree
}

fn internal_edge_count(graph: &Graph, partition: &Partition, nodes: &[NodeId]) -> f64 {
    let mut community_edges = 0.0;
    for &node in nodes {
        if let Some(neighbors) = graph.adjacency_list.get(&node) {
            for &neighbor in neighbors {
                if node < neighbor
                    && let Some(neighbor_comm) = partition.get(&neighbor)
                    && neighbor_comm == &partition[&node]
                {
                    community_edges += 1.0;
                }
            }
        }
    }
    community_edges
}

/// Shi's decomposed-modularity objectives (Shi et al. 2012, Eqs. 3.5/3.6),
/// both **minimized**:
///   `intra = 1 − Σ_c l_c/m`   (Eq. 3.5; `l_c` = internal edges counted once)
///   `inter = Σ_c (d_c/2m)^2`   (Eq. 3.6; `d_c` = Σ deg over c, each internal edge ×2)
/// so that modularity `Q = 1 − intra − inter`.
pub fn calculate_objectives(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<NodeId, usize>,
    parallel: bool,
) -> Metrics {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return Metrics::default();
    }

    let mut communities: HashMap<i32, Vec<NodeId>> = HashMap::default();
    for (&node, &comm) in partition {
        communities.entry(comm).or_default().push(node);
    }

    let total_edges_doubled = 2.0 * total_edges;

    let folder = |(mut intra_acc, mut inter_acc), (_, nodes): (&i32, &Vec<NodeId>)| {
        let community_degree = degree_sum(nodes, degrees);
        let community_edges = internal_edge_count(graph, partition, nodes);

        intra_acc += community_edges;
        inter_acc += (community_degree / total_edges_doubled).powi(2);
        (intra_acc, inter_acc)
    };

    let (intra_sum, inter) = if parallel && communities.len() > PARALLEL_COMMUNITY_THRESHOLD {
        communities
            .par_iter()
            .fold(|| (0.0, 0.0), folder)
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        communities.iter().fold((0.0, 0.0), folder)
    };

    let intra = 1.0 - (intra_sum / total_edges);

    Metrics { intra, inter }
}
