//! Locus-based adjacency genome (Park & Song 1989, as used by Pizzuti's
//! GA-Net / MOGA-Net): an individual is `N` genes `g_1..g_N`, one per node.
//! "A value `j` assigned to the ith gene is interpreted as a link between
//! nodes `i` and `j`" -- decoding unions every gene-edge and the connected
//! components are the communities (linear-time union-find, no fixed
//! community count). Own copy local to MOGA-Net (not shared with any other
//! detector's locus module) so this engine is fully self-contained.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use rand::{Rng, RngExt}; // rand 0.10: random_range lives on RngExt
use rustc_hash::FxHashMap;

pub type Genome = Vec<NodeId>;

/// Precomputed locus bookkeeping for one graph: a stable node ordering, the
/// reverse `NodeId -> position` map (used by `decode`'s union-find), and
/// per-position candidate lists (`{node itself} ∪ neighbours`) -- the "safe"
/// allele set for gene `i` (Pizzuti 2009, Sec. 4: "the possible values an
/// allele can assume are restricted to the neighbors of gene i"; a node with
/// no neighbours can only take itself, which decodes to a singleton).
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
        Locus { nodes, index_of, candidates }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// "Biased"/safe initialization (Pizzuti 2009, Sec. 4): generate gene `i`
    /// directly as a uniformly random pick from `{i itself} ∪ neighbours(i)`.
    /// This is equivalent in end distribution to "generate a fully random
    /// individual, then repair any (i,j) that isn't a real edge by replacing
    /// j with a random neighbour of i" -- every individual built this way is
    /// safe by construction, so no separate repair pass is needed anywhere
    /// else in the pipeline.
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
