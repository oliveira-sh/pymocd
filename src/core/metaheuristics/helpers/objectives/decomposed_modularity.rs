//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::objectives::metrics::Metrics;
use rayon::iter::*;
use rustc_hash::FxHashMap as HashMap;

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
    for (&node, &comm) in partition.iter() {
        communities.entry(comm).or_default().push(node);
    }

    let total_edges_doubled = 2.0 * total_edges;

    let folder = |(mut intra_acc, mut inter_acc), (_, nodes): (&i32, &Vec<NodeId>)| {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;
        for &node in nodes {
            let degree = *degrees.get(&node).unwrap_or(&0) as f64;
            community_degree += degree;
        }
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

        intra_acc += community_edges;
        inter_acc += (community_degree / total_edges_doubled).powi(2);
        (intra_acc, inter_acc)
    };

    let (intra_sum, inter) = if parallel && communities.len() > 8 {
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
