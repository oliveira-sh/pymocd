//! Locus-based adjacency genome (Park & Song 1998) and its decoding.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};
use std::collections::HashMap;

use crate::core::graph::{Graph, NodeId};

/// Gene `p` holds a *position* in `0..n-1`: "link position `p` to that position".
pub type Genome = Vec<usize>;

/// Position -> `NodeId` (used only at the module boundary) plus per-position
/// neighbour lists, which are both the allele domain and the adjacency.
pub struct Locus {
    pub nodes: Vec<NodeId>,
    pub neighbors: Vec<Vec<usize>>,
}

impl Locus {
    pub fn build(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone();
        let position_of: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(p, &v)| (v, p)).collect();
        let neighbors: Vec<Vec<usize>> = nodes
            .iter()
            .map(|v| graph.neighbors(v).iter().map(|u| position_of[u]).collect())
            .collect();
        Self { nodes, neighbors }
    }

    #[inline]
    pub const fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Safe allele for position `p` (Pizzuti 2009, Sec. 4): uniform over
    /// neighbours(p). The self-allele exists only for isolated nodes.
    #[inline]
    pub fn random_allele(&self, p: usize, rng: &mut impl Rng) -> usize {
        let nb = &self.neighbors[p];
        if nb.is_empty() {
            p
        } else {
            nb[rng.random_range(0..nb.len())]
        }
    }

    /// Safe initialization: every gene is a safe allele by construction, so no
    /// repair pass is needed anywhere in the pipeline.
    pub fn random_genome(&self, rng: &mut impl Rng) -> Genome {
        (0..self.n()).map(|p| self.random_allele(p, rng)).collect()
    }

    /// Union-find decode to compact labels indexed by position. Labels are
    /// assigned in ascending-position first-visit order, so equal partitions
    /// always get identical label arrays.
    pub fn decode(&self, genome: &Genome) -> Vec<i32> {
        let n = self.n();
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        for (p, &q) in genome.iter().enumerate() {
            union(&mut parent, p, q);
        }

        let mut labels = vec![0i32; n];
        let mut root_label = vec![-1i32; n];
        let mut next = 0i32;
        for (p, label) in labels.iter_mut().enumerate() {
            let root = find(&mut parent, p);
            if root_label[root] < 0 {
                root_label[root] = next;
                next += 1;
            }
            *label = root_label[root];
        }
        labels
    }
}
