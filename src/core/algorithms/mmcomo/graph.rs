//! The index-space graph, and the conversions to and from caller node ids.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::collections::{HashMap, HashSet};

use crate::core::algorithms::mmcomo::Labels;

/// Undirected graph in contiguous index space `[0, n)`.
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<usize>>,
    pub deg: Vec<f64>,
    /// `2|E|`.
    pub m2: f64,
}

impl Graph {
    fn from_indexed(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for &(u, v) in edges {
            if u != v && u < n && v < n {
                sets[u].insert(v);
                sets[v].insert(u);
            }
        }
        let adj: Vec<Vec<usize>> = sets
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2 = deg.iter().sum();
        Self { n, adj, deg, m2 }
    }
}

/// Build the index-space graph plus the index-to-id table and the isolated-node mask.
pub fn build(nodes: &[i32], edges: &[(i32, i32)]) -> (Graph, Vec<i32>, Vec<bool>) {
    let mut ids: Vec<i32> = Vec::with_capacity(nodes.len() + 2 * edges.len());
    ids.extend_from_slice(nodes);
    for &(u, v) in edges {
        ids.push(u);
        ids.push(v);
    }
    ids.sort_unstable();
    ids.dedup();
    let index: HashMap<i32, usize> = ids.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let eidx: Vec<(usize, usize)> = edges.iter().map(|&(u, v)| (index[&u], index[&v])).collect();
    let g = Graph::from_indexed(ids.len(), &eidx);
    let isolated: Vec<bool> = g.deg.iter().map(|&d| d == 0.0).collect();
    (g, ids, isolated)
}

/// Map index-space labels to `(node_id, community)`: isolated nodes get `-1`,
/// the remaining community ids are renumbered to `0..k`.
pub fn to_output(labels: &Labels, ids: &[i32], isolated: &[bool]) -> Vec<(i32, i32)> {
    let mut remap: HashMap<i32, i32> = HashMap::new();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let comm = if isolated[i] {
            -1
        } else {
            *remap.entry(labels[i]).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        };
        out.push((ids[i], comm));
    }
    out
}
