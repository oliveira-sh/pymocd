//! Locus-based adjacency representation (Park & Song; used independently by
//! Shi et al. 2012 §3.1.3, Fig. 1 — the same scheme MOGA-Net's paper uses).
//! Gene `g_i` holds one of node `i`'s neighbours ("including node i itself",
//! which we use to make degree-0 nodes safe by construction). Decoding a
//! genome identifies connected components of the implied directed graph
//! `i -> g_i`; we do this with union-find, which is linear-time and produces
//! the exact same components as the paper's backtracking scheme.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use rand::RngExt;
use rustc_hash::FxHashMap;

/// Dense `[0, n)` index for a graph's NodeId set (NodeId is not guaranteed
/// contiguous — built from `graph.nodes_vec()`), plus, per dense index, the
/// list of dense-index alleles a locus gene may take (the node's neighbours,
/// or `[self]` for isolated nodes).
pub struct NodeIndex {
    pub index_to_node: Vec<NodeId>,
    // Kept alongside `index_to_node` as the dense<->NodeId inverse; only used
    // during `build` and by tests today, allowed dead in non-test builds.
    #[allow(dead_code)]
    pub node_to_index: FxHashMap<NodeId, usize>,
    pub neighbor_candidates: Vec<Vec<usize>>,
}

impl NodeIndex {
    pub fn build(graph: &Graph) -> Self {
        let index_to_node: Vec<NodeId> = graph.nodes_vec().clone();
        let node_to_index: FxHashMap<NodeId, usize> = index_to_node
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();

        let neighbor_candidates: Vec<Vec<usize>> = index_to_node
            .iter()
            .enumerate()
            .map(|(i, &node)| {
                let neighbors = graph.neighbors(&node);
                if neighbors.is_empty() {
                    // Degree-0 node: "each gi can take one of the adjacent
                    // nodes of node i (including node i itself)" — self-allele.
                    vec![i]
                } else {
                    neighbors.iter().map(|n| node_to_index[n]).collect()
                }
            })
            .collect();

        Self {
            index_to_node,
            node_to_index,
            neighbor_candidates,
        }
    }

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.index_to_node.len()
    }
}

/// gene i (dense index) -> allele (dense index of a neighbour, or self).
pub type Genome = Vec<usize>;

pub fn random_genome(idx: &NodeIndex, rng: &mut impl rand::Rng) -> Genome {
    (0..idx.n())
        .map(|i| {
            let cands = &idx.neighbor_candidates[i];
            cands[rng.random_range(0..cands.len())]
        })
        .collect()
}

/// Union-find with path halving (linear-time, equivalent to the paper's
/// backtracking component scheme).
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Decode a locus genome into a community `Partition`: communities are the
/// connected components of the implied graph `i -> genome[i]`.
pub fn decode(genome: &Genome, idx: &NodeIndex) -> Partition {
    let n = genome.len();
    let mut parent: Vec<usize> = (0..n).collect();

    for (i, &j) in genome.iter().enumerate() {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            parent[ri] = rj;
        }
    }

    let mut partition = Partition::default();
    for i in 0..n {
        let root = find(&mut parent, i);
        partition.insert(idx.index_to_node[i], root as i32);
    }
    partition
}

/// Shi's "uniform two-point crossover" — by the paper's own functional
/// description ("unbiased w.r.t. the ordering of genes, able to generate any
/// combination of alleles from the two parents") this is plain per-gene
/// uniform crossover, not classic two-segment crossover. Always valid: each
/// gene's allele is inherited verbatim from a parent, so it is always one of
/// that gene's legal alleles.
pub fn uniform_crossover(p1: &Genome, p2: &Genome, rng: &mut impl rand::Rng) -> Genome {
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
        .collect()
}

/// "we randomly select some genes and assign them with other randomly
/// selected adjacent nodes." Per-gene independent probability `p_m`; the
/// paper does not say whether the resample may repeat the current allele —
/// we allow it (uniform draw over the same candidate set used at init), the
/// simplest reading and consistent with `random_genome`.
pub fn mutate(genome: &mut Genome, idx: &NodeIndex, p_m: f64, rng: &mut impl rand::Rng) {
    for i in 0..genome.len() {
        if rng.random_bool(p_m) {
            let cands = &idx.neighbor_candidates[i];
            genome[i] = cands[rng.random_range(0..cands.len())];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Graph;

    #[test]
    fn decode_two_triangles_plus_bridge_can_split() {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        let idx = NodeIndex::build(&g);
        // Genome forming exactly two triangle components (no use of the bridge).
        let genome: Genome = vec![1, 2, 0, 4, 5, 3];
        let partition = decode(&genome, &idx);
        assert_eq!(partition[&0], partition[&1]);
        assert_eq!(partition[&1], partition[&2]);
        assert_eq!(partition[&3], partition[&4]);
        assert_eq!(partition[&4], partition[&5]);
        assert_ne!(partition[&0], partition[&3]);
    }

    #[test]
    fn isolated_node_gets_self_allele() {
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.nodes.insert(2); // isolated
        g.adjacency_list.entry(2).or_default();
        g.finalize();
        let idx = NodeIndex::build(&g);
        let iso_dense = idx.node_to_index[&2];
        assert_eq!(idx.neighbor_candidates[iso_dense], vec![iso_dense]);
    }
}
