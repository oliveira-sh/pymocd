//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
//! Shi's decomposed-modularity objectives and their value type.

use super::locus::NodeIndex;
use crate::core::graph::Graph;

/// Shi's decomposed-modularity objectives (Shi et al. 2012, Eqs. 3.5/3.6),
/// both **minimized**:
///   `intra = 1 − Σ_c l_c/m`   (Eq. 3.5; `l_c` = internal edges counted once)
///   `inter = Σ_c (d_c/2m)^2`   (Eq. 3.6; `d_c` = Σ deg over c, each internal edge ×2)
/// so that modularity `Q = 1 − intra − inter`. `labels` are compacted dense
/// community ids and `degrees` node degrees, both indexed by node position.
pub fn calculate_objectives(
    graph: &Graph,
    idx: &NodeIndex,
    labels: &[i32],
    degrees: &[usize],
) -> Metrics {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return Metrics::default();
    }

    let num_comms = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut community_edges = vec![0.0f64; num_comms];
    let mut community_degree = vec![0.0f64; num_comms];

    for pos in 0..labels.len() {
        let comm = labels[pos] as usize;
        community_degree[comm] += degrees[pos] as f64;
        // `neighbor_candidates` = real neighbours in dense positions; the
        // isolated-node self-allele fails the pos < neighbor guard.
        for &neighbor in &idx.neighbor_candidates[pos] {
            if pos < neighbor && labels[neighbor] == labels[pos] {
                community_edges[comm] += 1.0;
            }
        }
    }

    let total_edges_doubled = 2.0 * total_edges;
    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    for comm in 0..num_comms {
        intra_sum += community_edges[comm];
        inter += (community_degree[comm] / total_edges_doubled).powi(2);
    }

    let intra = 1.0 - (intra_sum / total_edges);

    Metrics { intra, inter }
}

#[derive(Debug, Default, PartialEq)]
pub struct Metrics {
    pub intra: f64,
    pub inter: f64,
}
