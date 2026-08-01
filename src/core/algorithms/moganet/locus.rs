//! Locus-based adjacency genome (Park & Song 1989, as used by Pizzuti's
//! GA-Net / MOGA-Net): gene `i` holding value `j` is a link between nodes
//! `i` and `j`; decoding unions every gene-edge and the connected components
//! are the communities (no fixed community count). Local copy so the engine
//! is fully self-contained.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId};
use rand::{Rng, RngExt}; // rand 0.10: random_range lives on RngExt
use std::collections::HashMap;

/// Position-valued alleles: gene `p` holds a *position* in `0..n-1`.
pub type Genome = Vec<usize>;

/// Precomputed locus bookkeeping: stable node ordering (position -> `NodeId`,
/// used only at the module boundary) and per-position neighbour-position
/// lists (allele domain + adjacency for evaluation).
pub struct Locus {
    pub nodes: Vec<NodeId>,
    pub neighbors: Vec<Vec<usize>>,
}

impl Locus {
    pub fn build(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone();
        // NodeId -> position boundary map: built once per run (cold path).
        let index_of: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(p, &v)| (v, p)).collect();
        let neighbors: Vec<Vec<usize>> = nodes
            .iter()
            .map(|v| graph.neighbors(v).iter().map(|u| index_of[u]).collect())
            .collect();
        Locus { nodes, neighbors }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Safe allele for position `p` (Pizzuti 2009, Sec. 4): a uniform pick
    /// from neighbours(p); the self-allele exists only for isolated nodes
    /// (the paper's repair maps invalid genes to "one of the neighbors of i").
    #[inline]
    pub fn random_allele(&self, p: usize, rng: &mut impl Rng) -> usize {
        let nb = &self.neighbors[p];
        if nb.is_empty() {
            p
        } else {
            nb[rng.random_range(0..nb.len())]
        }
    }

    /// "Biased"/safe initialization (Pizzuti 2009, Sec. 4): every gene is a
    /// safe allele by construction, so no repair pass is needed anywhere in
    /// the pipeline.
    pub fn random_genome(&self, rng: &mut impl Rng) -> Genome {
        (0..self.n()).map(|p| self.random_allele(p, rng)).collect()
    }

    /// Decode genome -> compact community labels indexed by position, via
    /// union-find over positions (labels assigned in ascending-position
    /// first-visit order, so equal partitions get identical label arrays).
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
