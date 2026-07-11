//! Dense CSR (Compressed Sparse Row) graph.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use rustc_hash::FxHashMap;


/// Compressed sparse row undirected graph over dense ids `[0, n)`.
pub struct CsrGraph {
    pub n: usize,
    pub m: usize,
    /// Row offsets, length `n + 1`. Neighbors of `u` are `adj[xadj[u]..xadj[u+1]]`.
    pub xadj: Vec<u32>,
    /// Concatenated neighbor lists, length `2m`.
    pub adj: Vec<u32>,
    /// Per-node degree, length `n`.
    pub deg: Vec<u32>,
    /// Each undirected edge once with `u < v`, length `m`. Drives the O(m)
    /// intra-edge count in the objective.
    pub edges: Vec<(u32, u32)>,
    /// Dense id -> original file id, for output / NMI scoring.
    pub labels: Vec<i32>,
}

impl CsrGraph {
    /// Build a dense CSR graph from a node list and edge list (in-memory / Python
    /// ingestion). Ids are interned in `nodes` order first so isolated nodes get
    /// dense ids; edges then add the neighbor lists. Originals kept in `labels`.
    /// Each undirected edge should appear once (both directions are added here).
    pub fn from_edges(nodes: &[i32], edges: &[(i32, i32)]) -> Self {
        let mut id_map: FxHashMap<i32, u32> = FxHashMap::default();
        let mut labels: Vec<i32> = Vec::new();
        let intern = |raw: i32, map: &mut FxHashMap<i32, u32>, labels: &mut Vec<i32>| -> u32 {
            *map.entry(raw).or_insert_with(|| {
                let d = labels.len() as u32;
                labels.push(raw);
                d
            })
        };
        for &nd in nodes {
            intern(nd, &mut id_map, &mut labels);
        }
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); labels.len()];
        for &(a, b) in edges {
            let du = intern(a, &mut id_map, &mut labels);
            let dv = intern(b, &mut id_map, &mut labels);
            let need = du.max(dv) as usize;
            if rows.len() <= need {
                rows.resize(need + 1, Vec::new());
            }
            if du != dv {
                rows[du as usize].push(dv);
                rows[dv as usize].push(du);
            }
        }
        Self::build_csr(labels, rows)
    }

    /// Lay out CSR (`xadj`/`adj`/`deg`) + unique-edge arrays from interned
    /// `labels` and per-node neighbor lists `rows`. Each undirected edge is
    /// present in `rows` from both endpoints; the `u < v` filter keeps one copy.
    fn build_csr(labels: Vec<i32>, mut rows: Vec<Vec<u32>>) -> Self {
        let n = labels.len();
        rows.resize(n, Vec::new());

        let mut xadj = vec![0u32; n + 1];
        for u in 0..n {
            xadj[u + 1] = xadj[u] + rows[u].len() as u32;
        }
        let mut adj = vec![0u32; xadj[n] as usize];
        let mut deg = vec![0u32; n];
        for u in 0..n {
            let start = xadj[u] as usize;
            let row = &rows[u];
            adj[start..start + row.len()].copy_from_slice(row);
            deg[u] = row.len() as u32;
        }

        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(adj.len() / 2);
        for u in 0..n as u32 {
            for &v in &rows[u as usize] {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
        let m = edges.len();

        CsrGraph {
            n,
            m,
            xadj,
            adj,
            deg,
            edges,
            labels,
        }
    }

    #[inline(always)]
    pub fn neighbors(&self, u: usize) -> &[u32] {
        &self.adj[self.xadj[u] as usize..self.xadj[u + 1] as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn builds_csr_from_edges() {
        // 0-1, 0-2, 1-2 triangle; self-loop on 2 dropped; isolated node 3 kept.
        let g = CsrGraph::from_edges(&[0, 1, 2, 3], &[(0, 1), (0, 2), (1, 2), (2, 2)]);
        assert_eq!(g.n, 4);
        assert_eq!(g.m, 3);
        assert_eq!(g.deg[2], 2);
        assert_eq!(g.deg[3], 0);
        let mut nb = g.neighbors(0).to_vec();
        nb.sort_unstable();
        assert_eq!(nb, vec![1, 2]);
    }
}
