//! Locus-based (Pizzuti GA-Net style) genome for NSGA-III-CCM, and the label
//! array it decodes to.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId};
use rand::{Rng, RngExt};

/// Cell `p` holds `p` itself or the position of one of `p`'s neighbours, so
/// every genome the operators produce is valid by construction: no repair step.
pub type Genome = Vec<usize>;

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

/// Uniform over `{p} ∪ neighbours(p)`: the extra index in the `0..=len` draw is
/// the self slot.
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

pub fn random_genome(neighbor_pos: &[Vec<usize>], rng: &mut impl Rng) -> Genome {
    (0..neighbor_pos.len())
        .map(|p| pick_cell(neighbor_pos, p, rng))
        .collect()
}

/// Union-find over positions: one community per connected component, labelled
/// by its root *position*, so every label is an index in `0..n` and the
/// objectives can accumulate into flat `n`-sized arrays.
pub fn decode(genome: &Genome) -> Vec<i32> {
    let n = genome.len();
    let mut uf = UnionFind::new(n);
    for (p, &q) in genome.iter().enumerate() {
        uf.union(p, q);
    }
    (0..n).map(|p| uf.find(p) as i32).collect()
}

/// Relabel by first-seen order so permutation-equivalent partitions compare equal.
pub fn canonical_labels(labels: &[i32]) -> Vec<i32> {
    let mut next_id = 0i32;
    let mut remap = vec![-1i32; labels.len()]; // labels are root positions in 0..n
    labels
        .iter()
        .map(|&c| {
            if remap[c as usize] == -1 {
                remap[c as usize] = next_id;
                next_id += 1;
            }
            remap[c as usize]
        })
        .collect()
}

pub fn uniform_crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
    a.iter()
        .zip(b.iter())
        .map(|(&ga, &gb)| if rng.random_bool(0.5) { ga } else { gb })
        .collect()
}

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
    use crate::core::algorithms::ccm::fixtures::two_triangles;

    #[test]
    fn decode_groups_by_component() {
        // each cell points at the next position within its own triangle
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

    #[test]
    fn canonical_labels_first_seen_order() {
        assert_eq!(canonical_labels(&[3, 3, 0, 0, 3]), vec![0, 0, 1, 1, 0]);
    }
}
