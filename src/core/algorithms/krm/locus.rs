//! Locus-based genome (Pizzuti GA-Net style, used by Shaik, Ravi & Deb's
//! NSGA-III-KRM): a `Vec<usize>` of node *positions* where cell `p` MUST hold
//! `p` itself or the position of one of `nodes[p]`'s neighbours; decoding
//! unions positions into communities. Many distinct genomes can encode one
//! partition -- see `canonical_labels`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, NodeId};
use rand::{Rng, RngExt}; // rand 0.10: random_range lives on RngExt
use std::collections::HashMap;

pub type Genome = Vec<usize>;

/// Precomputed locus bookkeeping for one graph: stable node ordering
/// (`position -> NodeId`, the module's only boundary map) and per-position
/// candidate lists (`{position itself} ∪ neighbour positions`).
pub struct Locus {
    pub nodes: Vec<NodeId>,
    pub candidates: Vec<Vec<usize>>,
}

impl Locus {
    pub fn build(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone();
        // NodeId -> position map, needed only while building (cold path).
        let index_of: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(p, &v)| (v, p)).collect();
        let candidates: Vec<Vec<usize>> = nodes
            .iter()
            .enumerate()
            .map(|(p, &v)| {
                let mut c = Vec::with_capacity(graph.degree(&v) + 1);
                c.push(p);
                c.extend(graph.neighbors(&v).iter().map(|u| index_of[u]));
                c
            })
            .collect();
        Self { nodes, candidates }
    }

    #[inline]
    pub const fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Random genome: each cell independently uniform over its candidates.
    pub fn random_genome(&self, rng: &mut impl Rng) -> Genome {
        self.candidates
            .iter()
            .map(|c| c[rng.random_range(0..c.len())])
            .collect()
    }

    /// Decode genome -> per-position community labels via union-find over
    /// positions: for each position `p`, union `p` with `genome[p]`. Labels
    /// are union-find roots: arbitrary ids `< n`, not compacted.
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

        (0..n).map(|p| find(&mut parent, p) as i32).collect()
    }

    /// Permutation-invariant label vector (communities relabeled by first-seen
    /// position order); used by the duplicate-permutation filter.
    pub fn canonical_labels(&self, labels: &[i32]) -> Vec<i32> {
        let mut remap = vec![-1i32; labels.len()]; // raw labels are roots < n
        let mut next = 0i32;
        labels
            .iter()
            .map(|&c| {
                let slot = &mut remap[c as usize];
                if *slot < 0 {
                    *slot = next;
                    next += 1;
                }
                *slot
            })
            .collect()
    }

    /// True iff `labels` puts every node into a single community.
    pub fn is_single_community(&self, labels: &[i32]) -> bool {
        labels.windows(2).all(|w| w[0] == w[1])
    }
}
