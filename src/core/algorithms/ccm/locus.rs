//! Locus-based (Pizzuti GA-Net style) genome for NSGA-III-CCM: a `Vec<NodeId>`
//! indexed by position in the stable node ordering `nodes`. Cell `p` always
//! holds `nodes[p]` itself or one of its neighbours, so every genome the
//! operators produce is valid by construction — no repair step needed.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rand::{Rng, RngExt};
use rustc_hash::FxHashMap;

pub type Genome = Vec<NodeId>;

/// `NodeId -> position` lookup for the stable ordering `nodes`.
pub fn build_index(nodes: &[NodeId]) -> FxHashMap<NodeId, usize> {
    nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, p))
        .collect()
}

/// Uniformly pick a value for `node`'s locus cell from `{node} ∪ neighbours(node)`.
/// Degree-0 nodes have no choice but themselves.
#[inline]
fn pick_cell(graph: &Graph, node: NodeId, rng: &mut impl Rng) -> NodeId {
    let neighbors = graph.neighbors(&node);
    if neighbors.is_empty() {
        node
    } else {
        let k = rng.random_range(0..=neighbors.len());
        if k == neighbors.len() {
            node
        } else {
            neighbors[k]
        }
    }
}

/// Random genome: every cell independently uniform over `{node} ∪ neighbours(node)`.
pub fn random_genome(graph: &Graph, nodes: &[NodeId], rng: &mut impl Rng) -> Genome {
    nodes
        .iter()
        .map(|&node| pick_cell(graph, node, rng))
        .collect()
}

/// Decode a locus genome into a label `Partition` by union-find over
/// *positions*; each connected component is one community, labelled by its
/// union-find root position. Isolated nodes decode to singletons
/// (`normalize_community_ids` forces them to `-1` later).
pub fn decode(nodes: &[NodeId], index_of: &FxHashMap<NodeId, usize>, genome: &Genome) -> Partition {
    let n = nodes.len();
    let mut uf = UnionFind::new(n);
    for (p, &v) in genome.iter().enumerate() {
        let q = index_of[&v];
        uf.union(p, q);
    }
    nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, uf.find(p) as CommunityId))
        .collect()
}

/// Uniform locus-respecting crossover: each gene taken from parent `a` or `b`
/// independently with 50/50 odds.
pub fn uniform_crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
    a.iter()
        .zip(b.iter())
        .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
        .collect()
}

/// Adjacency-constrained mutation: each gene independently resampled (with
/// probability `mut_rate`) from `{node} ∪ neighbours(node)`.
pub fn mutate(
    genome: &mut Genome,
    graph: &Graph,
    nodes: &[NodeId],
    mut_rate: f64,
    rng: &mut impl Rng,
) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            *gene = pick_cell(graph, nodes[p], rng);
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn decode_groups_by_component() {
        let g = two_triangles();
        let nodes = g.nodes_vec().clone();
        let index_of = build_index(&nodes);
        // Each cell points at the next node within its own triangle.
        let genome: Genome = vec![1, 2, 0, 4, 5, 3];
        let part = decode(&nodes, &index_of, &genome);
        assert_eq!(part[&0], part[&1]);
        assert_eq!(part[&1], part[&2]);
        assert_eq!(part[&3], part[&4]);
        assert_eq!(part[&4], part[&5]);
        assert_ne!(part[&0], part[&3]);
    }

    #[test]
    fn random_genome_is_always_valid() {
        let g = two_triangles();
        let nodes = g.nodes_vec().clone();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let genome = random_genome(&g, &nodes, &mut rng);
            for (p, &v) in genome.iter().enumerate() {
                let node = nodes[p];
                assert!(v == node || g.neighbors(&node).contains(&v));
            }
        }
    }
}
