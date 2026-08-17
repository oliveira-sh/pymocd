//! Sec. 4.3.1's lighter chromosome: the community relation graph of Fig. 6.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::algorithms::cdrme::topology::Topology;

/// One gene per community instead of one per node: `inner[c]` is the loop
/// weight of Fig. 6 (`innerLinks`), `adj[c][d]` the link weight
/// (`outerLinks(c,d)`) and `total[c]` is `totalLinks(c)`, the links with at
/// least one end in `c`.
pub struct Relation {
    pub inner: Vec<u32>,
    pub total: Vec<u32>,
    pub adj: Vec<FxHashMap<u32, u32>>,
    pub live: Vec<u32>,
    position: Vec<usize>,
}

impl Relation {
    pub fn from_labels(topology: &Topology, labels: &[u32], k: usize) -> Self {
        let mut inner = vec![0u32; k];
        let mut adj: Vec<FxHashMap<u32, u32>> = (0..k).map(|_| FxHashMap::default()).collect();
        for &u in &topology.active {
            for &v in topology.neighbors(u) {
                if u >= v {
                    continue;
                }
                let (cu, cv) = (labels[u as usize], labels[v as usize]);
                if cu == cv {
                    inner[cu as usize] += 1;
                } else {
                    *adj[cu as usize].entry(cv).or_insert(0) += 1;
                    *adj[cv as usize].entry(cu).or_insert(0) += 1;
                }
            }
        }
        let total = (0..k)
            .map(|c| inner[c] + adj[c].values().sum::<u32>())
            .collect();

        Self {
            inner,
            total,
            adj,
            live: (0..k as u32).collect(),
            position: (0..k).collect(),
        }
    }

    pub const fn len(&self) -> usize {
        self.live.len()
    }

    /// The neighbour communities of `c`, in ascending id order so the uniform
    /// draw of Sec. 4.3.2 never depends on hash order.
    pub fn neighbors(&self, c: u32) -> Vec<u32> {
        let mut neighbors: Vec<u32> = self.adj[c as usize].keys().copied().collect();
        neighbors.sort_unstable();
        neighbors
    }

    /// Folds `j` into `i`; `j` stops being live.
    pub fn merge(&mut self, i: u32, j: u32) {
        let bridge = self.adj[i as usize].remove(&j).unwrap_or(0);
        self.adj[j as usize].remove(&i);
        self.inner[i as usize] += self.inner[j as usize] + bridge;
        self.total[i as usize] = self.total[i as usize] + self.total[j as usize] - bridge;

        let moved: Vec<(u32, u32)> = self.adj[j as usize].drain().collect();
        for (c, w) in moved {
            self.adj[c as usize].remove(&j);
            *self.adj[c as usize].entry(i).or_insert(0) += w;
            *self.adj[i as usize].entry(c).or_insert(0) += w;
        }
        self.inner[j as usize] = 0;
        self.total[j as usize] = 0;

        let slot = self.position[j as usize];
        let moved_up = *self.live.last().unwrap();
        self.live.swap_remove(slot);
        self.position[moved_up as usize] = slot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // three triangles bridged 2-3 and 5-6
    fn chain() -> (Topology, Vec<u32>) {
        let nodes: Vec<i32> = (0..9).collect();
        let edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (6, 7),
            (7, 8),
            (6, 8),
            (2, 3),
            (5, 6),
        ];
        let topology = Topology::from_edges(&nodes, &edges);
        (topology, vec![0, 0, 0, 1, 1, 1, 2, 2, 2])
    }

    #[test]
    fn the_relation_graph_counts_inner_outer_and_total_links() {
        let (topology, labels) = chain();
        let relation = Relation::from_labels(&topology, &labels, 3);
        assert_eq!(relation.inner, vec![3, 3, 3]);
        assert_eq!(relation.total, vec![4, 5, 4]);
        assert_eq!(relation.adj[1][&0], 1);
        assert_eq!(relation.adj[1][&2], 1);
        assert_eq!(relation.len(), 3);
        assert_eq!(relation.neighbors(1), vec![0, 2]);
    }

    #[test]
    fn merging_folds_the_bridge_into_the_inner_links() {
        let (topology, labels) = chain();
        let mut relation = Relation::from_labels(&topology, &labels, 3);
        relation.merge(0, 1);
        assert_eq!(relation.len(), 2);
        assert_eq!(relation.live, vec![0, 2]);
        assert_eq!(relation.inner[0], 3 + 3 + 1);
        assert_eq!(relation.total[0], 4 + 5 - 1);
        assert_eq!(relation.adj[0][&2], 1);
        assert_eq!(relation.adj[2][&0], 1);
        assert!(!relation.adj[2].contains_key(&1));
    }
}
