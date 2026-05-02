//! Dense view (`DenseGraph`) over a sparse `Graph`. Built once per envolve.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::EdgeUV;
use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::cmp::Ordering;

pub struct DenseGraph {
    pub n: usize,
    pub total_edges: usize,
    pub degrees: Vec<u32>,
    /// CSR offsets. Neighbors of i: adj_flat[adj_starts[i] .. adj_starts[i+1]].
    pub adj_starts: Vec<u32>,
    pub adj_flat: Vec<u32>,
    /// SoA edges. Each undirected edge once with u < v.
    pub edge_uv: Vec<EdgeUV>,
    pub edge_w: Vec<f64>,
    pub nodes: Vec<NodeId>,
    /// Sum of incident edge weights per node.
    pub weighted_deg: Vec<f64>,
    pub total_weight: f64,
}

impl DenseGraph {
    pub fn from_graph(g: &Graph) -> Self {
        let nodes: Vec<NodeId> = g.nodes_vec().clone();
        let n = nodes.len();
        let mut node_to_idx: FxHashMap<NodeId, u32> =
            FxHashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for (i, &nid) in nodes.iter().enumerate() {
            node_to_idx.insert(nid, i as u32);
        }

        let mut nb_tmp: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut degrees: Vec<u32> = Vec::with_capacity(n);
        let mut total_neighbors: usize = 0;
        for &nid in &nodes {
            let neighbors = g.neighbors(&nid);
            let mut dense_nb: Vec<u32> = Vec::with_capacity(neighbors.len());
            for &nb in neighbors {
                if let Some(&j) = node_to_idx.get(&nb) {
                    dense_nb.push(j);
                }
            }
            dense_nb.sort_unstable();
            degrees.push(dense_nb.len() as u32);
            total_neighbors += dense_nb.len();
            nb_tmp.push(dense_nb);
        }

        let mut adj_starts: Vec<u32> = Vec::with_capacity(n + 1);
        let mut adj_flat: Vec<u32> = Vec::with_capacity(total_neighbors);
        adj_starts.push(0);
        for nb in &nb_tmp {
            adj_flat.extend_from_slice(nb);
            adj_starts.push(adj_flat.len() as u32);
        }

        let mut edges_uv: Vec<(u32, u32)> = Vec::with_capacity(g.num_edges());
        for i in 0..n {
            let iu = i as u32;
            for &j in &nb_tmp[i] {
                if j > iu {
                    edges_uv.push((iu, j));
                }
            }
        }

        // TOM/Jaccard edge weights via two-pointer intersection on sorted CSR.
        let mut edge_uv: Vec<EdgeUV> = Vec::with_capacity(edges_uv.len());
        let mut edge_w: Vec<f64> = Vec::with_capacity(edges_uv.len());
        let mut weighted_deg: Vec<f64> = vec![0.0; n];
        for &(u, v) in &edges_uv {
            let a = &nb_tmp[u as usize];
            let b = &nb_tmp[v as usize];
            let mut i = 0usize;
            let mut j = 0usize;
            let mut inter: u32 = 0;
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    Ordering::Equal => {
                        inter += 1;
                        i += 1;
                        j += 1;
                    }
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                }
            }
            let aug_inter = (inter + 2) as f64;
            let union = (a.len() + b.len()) as f64 - inter as f64;
            let w = (aug_inter / union.max(1.0)).max(1e-3);
            edge_uv.push((u as u64) | ((v as u64) << 32));
            edge_w.push(w);
            weighted_deg[u as usize] += w;
            weighted_deg[v as usize] += w;
        }
        let total_weight: f64 = edge_w.iter().copied().sum();

        DenseGraph {
            n,
            total_edges: g.num_edges(),
            degrees,
            adj_starts,
            adj_flat,
            edge_uv,
            edge_w,
            nodes,
            weighted_deg,
            total_weight,
        }
    }

    #[inline(always)]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let s = self.adj_starts[i] as usize;
        let e = self.adj_starts[i + 1] as usize;
        &self.adj_flat[s..e]
    }

    pub fn sparse_partition(&self, dense: &[CommunityId]) -> Partition {
        let mut p = Partition::default();
        p.reserve(self.n);
        for (i, &c) in dense.iter().enumerate() {
            p.insert(self.nodes[i], c);
        }
        p
    }
}
