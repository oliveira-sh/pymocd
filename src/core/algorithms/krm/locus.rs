//! The locus genome and the per-graph bookkeeping it is decoded against.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};
use std::collections::HashMap;

use crate::core::graph::{Graph, NodeId};

/// Cell `p` holds `p` itself or the position of one of `nodes[p]`'s neighbours.
/// Every operator preserves that, so no repair pass exists anywhere.
pub type Genome = Vec<usize>;

/// `nodes` maps a position to a `NodeId`; `candidates[p]` is `p` itself
/// followed by the positions of `nodes[p]`'s neighbours, in that order.
pub struct Locus {
    pub nodes: Vec<NodeId>,
    pub candidates: Vec<Vec<usize>>,
}

impl Locus {
    pub fn build(graph: &Graph) -> Self {
        let nodes = graph.nodes_vec().clone();
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

    pub fn random_genome(&self, rng: &mut impl Rng) -> Genome {
        self.candidates
            .iter()
            .map(|c| c[rng.random_range(0..c.len())])
            .collect()
    }

    /// Per-position labels, by union-find of `p` with `genome[p]`. A label is a
    /// component root: an arbitrary id `< n`, not compacted, which is what lets
    /// callers index flat `n`-sized arrays by it.
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

    /// Communities relabelled by first-seen position order, so permutations of
    /// one partition compare equal.
    pub fn canonical_labels(&self, labels: &[i32]) -> Vec<i32> {
        let mut remap = vec![-1i32; labels.len()];
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

    pub fn is_single_community(&self, labels: &[i32]) -> bool {
        labels.windows(2).all(|w| w[0] == w[1])
    }
}
