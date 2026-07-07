//! Locus-based adjacency genome (Park & Song 1989, as used by Pizzuti's
//! GA-Net / MOGA-Net): gene `i` holding value `j` is a link between nodes
//! `i` and `j`; decoding unions every gene-edge and the connected components
//! are the communities (no fixed community count). Local copy so the engine
//! is fully self-contained.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use rand::{Rng, RngExt}; // rand 0.10: random_range lives on RngExt
use rustc_hash::FxHashMap;

pub type Genome = Vec<NodeId>;

/// Precomputed locus bookkeeping: stable node ordering, reverse
/// `NodeId -> position` map, and per-position "safe" allele sets
/// (`{node itself} ∪ neighbours`, Pizzuti 2009, Sec. 4; a degree-0 node can
/// only take itself, decoding to a singleton).
pub struct Locus {
    pub nodes: Vec<NodeId>,
    pub index_of: FxHashMap<NodeId, usize>,
    pub candidates: Vec<Vec<NodeId>>,
}

impl Locus {
    pub fn build(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone();
        let index_of: FxHashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(p, &v)| (v, p)).collect();
        let candidates: Vec<Vec<NodeId>> = nodes
            .iter()
            .map(|&v| {
                let mut c = Vec::with_capacity(graph.degree(&v) + 1);
                c.push(v);
                c.extend_from_slice(graph.neighbors(&v));
                c
            })
            .collect();
        Locus {
            nodes,
            index_of,
            candidates,
        }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// "Biased"/safe initialization (Pizzuti 2009, Sec. 4): gene `i` is a
    /// uniform pick from `{i itself} ∪ neighbours(i)` -- safe by
    /// construction, so no repair pass is needed anywhere in the pipeline.
    pub fn random_genome(&self, rng: &mut impl Rng) -> Genome {
        self.candidates
            .iter()
            .map(|c| c[rng.random_range(0..c.len())])
            .collect()
    }

    /// Decode genome -> Partition via union-find over positions: for each
    /// position `p` holding value `v`, union `p` with `index_of[v]`.
    pub fn decode(&self, genome: &Genome) -> Partition {
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

        for (p, &v) in genome.iter().enumerate() {
            let q = self.index_of[&v];
            union(&mut parent, p, q);
        }

        let mut partition = Partition::default();
        for p in 0..n {
            let root = find(&mut parent, p);
            partition.insert(self.nodes[p], root as i32);
        }
        partition
    }
}
