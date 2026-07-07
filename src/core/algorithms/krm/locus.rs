//! Locus-based genome representation (Pizzuti GA-Net style, used by Shaik,
//! Ravi & Deb's NSGA-III-KRM). A genome is a `Vec<NodeId>` of length `n`: the
//! cell at position `p` (representing `nodes[p]`) holds a `NodeId` value that
//! MUST be either that node itself or one of its neighbours. Decoding unions
//! positions whose cell points at each other; each resulting connected
//! component is one community. A single partition can be encoded by many
//! distinct genomes (permutation-equivalent solutions) -- see
//! `canonical_labels`, used by the duplicate-permutation filter.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId, Partition};
use rand::{Rng, RngExt}; // rand 0.10: random_range lives on RngExt
use rustc_hash::FxHashMap;

pub type Genome = Vec<NodeId>;

/// Precomputed locus bookkeeping for one graph: the stable node ordering, the
/// reverse `NodeId -> position` map (needed by `decode`'s union-find), and
/// per-position candidate value lists (`{node itself} ∪ neighbours`) so
/// genome init/mutation never has to touch the graph's adjacency lists.
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

    /// Random genome: each cell independently uniform over `{node itself} ∪
    /// {its neighbours}` (degree-0 nodes have a single candidate, themselves).
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

    /// Canonicalize a decoded partition into a permutation-invariant label
    /// vector: communities are relabeled by first-seen order over `nodes`,
    /// so any two permutation-equivalent partitions produce the same vector.
    /// Used by the paper's duplicate-permutation filter.
    pub fn canonical_labels(&self, partition: &Partition) -> Vec<i32> {
        let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
        let mut next = 0i32;
        self.nodes
            .iter()
            .map(|node| {
                let c = *partition.get(node).expect("decode covers every node");
                *remap.entry(c).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect()
    }

    /// True iff `partition` puts every node into a single community (the
    /// solution the paper's second customization excludes).
    pub fn is_single_community(&self, partition: &Partition) -> bool {
        let mut first: Option<i32> = None;
        for node in &self.nodes {
            let c = *partition.get(node).expect("decode covers every node");
            match first {
                None => first = Some(c),
                Some(f) if f != c => return false,
                _ => {}
            }
        }
        true
    }
}
