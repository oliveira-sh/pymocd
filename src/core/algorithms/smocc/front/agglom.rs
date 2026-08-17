//! Agglomerative coarsening of a partition, emitted as a chain.
//!
//! The search returns communities that are pure but too many: homogeneity
//! stays high while completeness falls as mixing rises. Neither the selector
//! nor the front can fix that, because the front holds nothing coarser. This
//! builds the missing candidates by repeatedly merging the adjacent pair whose
//! merge most improves modularity, and by emitting the partition at several
//! points along the way rather than only at the end. The endpoint is where
//! modularity's resolution limit has finished merging, which is far past the
//! planted granularity; some level along the chain is close to it, and the
//! front's own dominance sorting keeps whichever levels are worth keeping.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::algorithms::smocc::Labels;
use crate::core::graph::CsrGraph;

/// How many levels of the merge chain to emit.
const LEVELS: usize = 8;

struct Dsu {
    parent: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
}

/// The community graph of a partition: each vertex's dense community index,
/// the number of communities, the inter-community edge weights keyed by the
/// ordered pair, and each community's volume.
struct CommunityGraph {
    dense: Vec<u32>,
    k: usize,
    between: FxHashMap<(u32, u32), f64>,
    vol: Vec<f64>,
}

/// Compress `part` to dense community indices and contract the graph over them.
fn community_graph(g: &CsrGraph, part: &[i32]) -> CommunityGraph {
    let n = g.n;
    let mut slot: FxHashMap<i32, u32> = FxHashMap::default();
    let mut dense = vec![0u32; n];
    for (u, &c) in part.iter().enumerate().take(n) {
        let next = slot.len() as u32;
        let id = *slot.entry(c).or_insert(next);
        dense[u] = id;
    }
    let k = slot.len();
    let mut vol = vec![0.0f64; k];
    for u in 0..n {
        vol[dense[u] as usize] += f64::from(g.deg[u]);
    }
    let mut between: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    for &(u, v) in &g.edges {
        let (a, b) = (dense[u as usize], dense[v as usize]);
        if a != b {
            let key = if a < b { (a, b) } else { (b, a) };
            *between.entry(key).or_insert(0.0) += 1.0;
        }
    }
    CommunityGraph {
        dense,
        k,
        between,
        vol,
    }
}

/// A chain of progressively coarser partitions derived from `part`.
///
/// Empty when no merge improves modularity, which is the common case once the
/// partition is already at or below the graph's modularity scale.
pub(super) fn agglomerate(g: &CsrGraph, part: &[i32]) -> Vec<Labels> {
    let m = g.m as f64;
    if m == 0.0 {
        return Vec::new();
    }
    let CommunityGraph {
        dense,
        k,
        between,
        vol,
    } = community_graph(g, part);
    if k < 3 {
        return Vec::new();
    }

    let two_m = 2.0 * m;
    let mut cand: Vec<((u32, u32), f64)> = between
        .into_iter()
        .map(|(key, w)| {
            let (a, b) = key;
            let gain = w / m - 2.0 * vol[a as usize] * vol[b as usize] / (two_m * two_m);
            (key, gain)
        })
        .filter(|&(_, gain)| gain > 0.0)
        .collect();
    if cand.is_empty() {
        return Vec::new();
    }
    // Descending gain, ties by the pair so the order never depends on the hash.
    cand.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });

    let mut dsu = Dsu::new(k);
    let mut vol_w = vol;
    let mut merges: Vec<(usize, usize)> = Vec::new();
    for ((a, b), _) in cand {
        let (mut r, mut s) = (dsu.find(a as usize), dsu.find(b as usize));
        if r == s {
            continue;
        }
        if r > s {
            std::mem::swap(&mut r, &mut s);
        }
        dsu.parent[s] = r;
        vol_w[r] += vol_w[s];
        merges.push((r, s));
    }
    if merges.is_empty() {
        return Vec::new();
    }

    let step = merges.len().div_ceil(LEVELS).max(1);
    let mut dsu = Dsu::new(k);
    let mut chain: Vec<Labels> = Vec::new();
    for (i, &(r, s)) in merges.iter().enumerate() {
        let (rr, ss) = (dsu.find(r), dsu.find(s));
        if rr != ss {
            dsu.parent[ss] = rr;
        }
        if (i + 1) % step == 0 || i + 1 == merges.len() {
            let root: Vec<i32> = (0..k).map(|c| dsu.find(c) as i32).collect();
            chain.push(dense.iter().map(|&c| root[c as usize]).collect());
        }
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cliques_split_in_four() -> (CsrGraph, Labels) {
        // Two 6-cliques joined by one edge, each clique cut in half by the
        // input partition: the merge chain should put each clique back.
        let nodes: Vec<i32> = (0..12).collect();
        let mut e = Vec::new();
        for base in [0, 6] {
            for a in base..base + 6 {
                for b in (a + 1)..base + 6 {
                    e.push((a, b));
                }
            }
        }
        e.push((5, 6));
        let g = CsrGraph::from_edges(&nodes, &e);
        let part: Labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        (g, part)
    }

    #[test]
    fn agglomerate_recovers_the_two_cliques() {
        let (g, part) = two_cliques_split_in_four();
        let chain = agglomerate(&g, &part);
        assert!(!chain.is_empty(), "no merge was accepted");
        let found = chain.iter().any(|p| {
            let mut u: Vec<i32> = p.clone();
            u.sort_unstable();
            u.dedup();
            u.len() == 2 && p[0] == p[5] && p[6] == p[11] && p[0] != p[6]
        });
        assert!(found, "the two cliques were never recovered: {chain:?}");
    }

    #[test]
    fn agglomerate_is_deterministic() {
        let (g, part) = two_cliques_split_in_four();
        assert_eq!(agglomerate(&g, &part), agglomerate(&g, &part));
    }

    #[test]
    fn agglomerate_declines_when_nothing_pays() {
        let (g, _) = two_cliques_split_in_four();
        let one: Labels = vec![0; g.n];
        assert!(agglomerate(&g, &one).is_empty());
    }
}
