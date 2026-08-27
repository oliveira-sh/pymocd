//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CsrGraph, NodeId};

pub struct Topology {
    pub n: usize,
    pub m: usize,
    pub labels: Vec<NodeId>,
    pub active: Vec<u32>,
    pub avg_degree: f64,
    pub enc: usize,
    xadj: Vec<u32>,
    adj: Vec<u32>,
}

impl Topology {
    pub fn from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> Self {
        let csr = CsrGraph::from_edges(nodes, edges);
        let n = csr.n;
        let mut xadj = vec![0u32; n + 1];
        let mut adj: Vec<u32> = Vec::with_capacity(csr.adj.len());
        let mut row: Vec<u32> = Vec::new();
        for u in 0..n {
            row.clear();
            row.extend_from_slice(csr.neighbors(u));
            row.sort_unstable();
            row.dedup();
            adj.extend_from_slice(&row);
            xadj[u + 1] = adj.len() as u32;
        }

        let active: Vec<u32> = (0..n as u32)
            .filter(|&u| xadj[u as usize + 1] > xadj[u as usize])
            .collect();
        let m = adj.len() / 2;
        let order = active.len() as f64;
        let avg_degree = if active.is_empty() {
            0.0
        } else {
            2.0 * m as f64 / order
        };
        let enc = if avg_degree > 0.0 {
            (order / avg_degree).round().clamp(1.0, order) as usize
        } else {
            0
        };

        Self {
            n,
            m,
            labels: csr.labels,
            active,
            avg_degree,
            enc,
            xadj,
            adj,
        }
    }

    #[inline]
    pub fn neighbors(&self, u: u32) -> &[u32] {
        &self.adj[self.xadj[u as usize] as usize..self.xadj[u as usize + 1] as usize]
    }

    #[inline]
    pub fn degree(&self, u: u32) -> usize {
        (self.xadj[u as usize + 1] - self.xadj[u as usize]) as usize
    }

    pub fn common_neighbors(&self, u: u32, v: u32) -> u32 {
        let (mut a, mut b) = (self.neighbors(u), self.neighbors(v));
        if a.len() > b.len() {
            std::mem::swap(&mut a, &mut b);
        }
        let mut count = 0;
        let mut j = 0;
        for &x in a {
            while j < b.len() && b[j] < x {
                j += 1;
            }
            if j == b.len() {
                break;
            }
            if b[j] == x {
                count += 1;
                j += 1;
            }
        }
        count
    }

    #[cfg(test)]
    pub fn similarity(&self, v: u32, u: u32) -> u32 {
        u32::from(self.neighbors(v).binary_search(&u).is_ok()) + self.common_neighbors(v, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_and_self_edges_collapse() {
        let t = Topology::from_edges(&[0, 1, 2, 3], &[(0, 1), (1, 0), (0, 1), (2, 2), (0, 2)]);
        assert_eq!(t.n, 4);
        assert_eq!(t.m, 2);
        assert_eq!(t.degree(0), 2);
        assert_eq!(t.degree(2), 1);
        assert_eq!(t.active, vec![0, 1, 2]);
    }

    #[test]
    fn similarity_is_connection_plus_common_neighbours() {
        let t = Topology::from_edges(&[0, 1, 2, 3], &[(0, 1), (1, 2), (0, 2), (2, 3)]);
        assert_eq!(t.similarity(0, 1), 1 + 1);
        assert_eq!(t.similarity(0, 3), 1);
        assert_eq!(t.similarity(1, 3), 1);
        assert_eq!(t.similarity(0, 1), t.similarity(1, 0));
    }

    #[test]
    fn enc_follows_equations_five_and_six() {
        let t = Topology::from_edges(&[0, 1, 2, 3], &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        assert!((t.avg_degree - 2.0).abs() < 1e-12);
        assert_eq!(t.enc, 2);
    }

    #[test]
    fn an_edgeless_graph_has_no_active_nodes() {
        let t = Topology::from_edges(&[0, 1, 2], &[]);
        assert!(t.active.is_empty());
        assert_eq!(t.enc, 0);
        assert_eq!(t.m, 0);
    }
}
