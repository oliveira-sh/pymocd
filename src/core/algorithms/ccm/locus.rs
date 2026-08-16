//! Locus-based (Pizzuti GA-Net style) genome for NSGA-III-CCM: a `Vec<usize>`
//! of node *positions* in the stable node ordering. Cell `p` always holds `p`
//! itself or the position of one of `p`'s neighbours, so every genome the
//! operators produce is valid by construction — no repair step needed.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId};
use rand::{Rng, RngExt};

pub type Genome = Vec<usize>;

/// Per-position neighbour candidates, built ONCE per run at the module
/// boundary: `neighbor_pos[p]` lists the positions of `nodes[p]`'s neighbours.
pub fn neighbor_positions(graph: &Graph, nodes: &[NodeId]) -> Vec<Vec<usize>> {
    let index_of: std::collections::HashMap<NodeId, usize> = nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, p))
        .collect();
    nodes
        .iter()
        .map(|node| graph.neighbors(node).iter().map(|v| index_of[v]).collect())
        .collect()
}

/// Uniformly pick a value for position `p`'s locus cell from
/// `{p} ∪ neighbour positions of p`. Degree-0 nodes have no choice but themselves.
#[inline]
fn pick_cell(neighbor_pos: &[Vec<usize>], p: usize, rng: &mut impl Rng) -> usize {
    let neighbors = &neighbor_pos[p];
    if neighbors.is_empty() {
        p
    } else {
        let k = rng.random_range(0..=neighbors.len());
        if k == neighbors.len() {
            p
        } else {
            neighbors[k]
        }
    }
}

/// Random genome: every cell independently uniform over `{p} ∪ neighbours(p)`.
pub fn random_genome(neighbor_pos: &[Vec<usize>], rng: &mut impl Rng) -> Genome {
    (0..neighbor_pos.len())
        .map(|p| pick_cell(neighbor_pos, p, rng))
        .collect()
}

/// Decode a locus genome into `Vec<i32>` labels indexed by position, by
/// union-find over positions; each connected component is one community,
/// labelled by its union-find root position. Isolated nodes decode to
/// singletons (`normalize_community_ids` forces them to `-1` later).
pub fn decode(genome: &Genome) -> Vec<i32> {
    let n = genome.len();
    let mut uf = UnionFind::new(n);
    for (p, &q) in genome.iter().enumerate() {
        uf.union(p, q);
    }
    (0..n).map(|p| uf.find(p) as i32).collect()
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
/// probability `mut_rate`) from `{p} ∪ neighbours(p)`.
pub fn mutate(genome: &mut Genome, neighbor_pos: &[Vec<usize>], mut_rate: f64, rng: &mut impl Rng) {
    for (p, gene) in genome.iter_mut().enumerate() {
        if rng.random_bool(mut_rate) {
            *gene = pick_cell(neighbor_pos, p, rng);
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
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
        // Each cell points at the next position within its own triangle.
        let genome: Genome = vec![1, 2, 0, 4, 5, 3];
        let labels = decode(&genome);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn random_genome_is_always_valid() {
        let g = two_triangles();
        let nodes = g.nodes_vec().clone();
        let neighbor_pos = neighbor_positions(&g, &nodes);
        let mut rng = rand::rng();
        for _ in 0..100 {
            let genome = random_genome(&neighbor_pos, &mut rng);
            for (p, &v) in genome.iter().enumerate() {
                assert!(v == p || neighbor_pos[p].contains(&v));
            }
        }
    }
}
