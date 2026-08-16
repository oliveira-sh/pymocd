//! Post-search front refinement: tiny-community reabsorption and disconnected
//! community splitting.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

use super::Labels;
use super::nsga2::fast_nondominated_sort;
use super::objectives::{ObjSet, evaluate};

fn refine_tiny(g: &CsrGraph, wadj: &[f64], part: &[i32], max_size: usize) -> Vec<i32> {
    let mut p = part.to_vec();
    for _ in 0..5 {
        let mut members: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
        for (u, &c) in p.iter().enumerate() {
            members.entry(c).or_default().push(u);
        }
        let tiny: Vec<i32> = members
            .iter()
            .filter(|(_, v)| v.len() <= max_size)
            .map(|(&c, _)| c)
            .collect();
        if tiny.is_empty() {
            break;
        }
        let mut moved = false;
        for c in tiny {
            let nodes = &members[&c];
            let mut internal = 0.0f64;
            let mut ext: FxHashMap<i32, f64> = FxHashMap::default();
            for &u in nodes {
                let start = g.xadj[u] as usize;
                let end = g.xadj[u + 1] as usize;
                for (&v, &w) in g.adj[start..end].iter().zip(&wadj[start..end]) {
                    let cv = p[v as usize];
                    if cv == c {
                        internal += w;
                    } else {
                        *ext.entry(cv).or_insert(0.0) += w;
                    }
                }
            }
            internal /= 2.0;
            let target = ext.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let sa = members.get(a.0).map_or(0, |v| v.len());
                        let sb = members.get(b.0).map_or(0, |v| v.len());
                        sa.cmp(&sb)
                    })
            });
            let Some((&tc, &te)) = target else { continue };
            if nodes.len() == 1 || internal == 0.0 || te > internal {
                for &u in nodes {
                    p[u] = tc;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    p
}

fn split_components(g: &CsrGraph, part: &[i32]) -> Option<Vec<i32>> {
    let n = g.n;
    let mut lab = vec![-1i32; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut k_new = 0usize;
    for s in 0..n {
        if lab[s] != -1 {
            continue;
        }
        k_new += 1;
        let c = part[s];
        lab[s] = s as i32;
        stack.push(s);
        while let Some(u) = stack.pop() {
            for &v in g.neighbors(u) {
                let v = v as usize;
                if lab[v] == -1 && part[v] == c {
                    lab[v] = s as i32;
                    stack.push(v);
                }
            }
        }
    }
    let k_old = part.iter().collect::<HashSet<_>>().len();
    (k_new > k_old).then_some(lab)
}

pub(super) fn refine_front(
    g: &CsrGraph,
    wadj: &[f64],
    front: Vec<Labels>,
    objset: ObjSet,
) -> Vec<Labels> {
    if front.is_empty() {
        return front;
    }
    let mut seen: HashSet<Vec<i32>> = front.iter().cloned().collect();
    let mut all: Vec<Labels> = front.clone();
    for p in &front {
        if let Some(split) = split_components(g, p) {
            let merged = refine_tiny(g, wadj, &split, 2);
            if seen.insert(merged.clone()) {
                all.push(merged);
            }
            if seen.insert(split.clone()) {
                all.push(split);
            }
        }
        let refined = refine_tiny(g, wadj, p, 2);
        if seen.insert(refined.clone()) {
            all.push(refined);
        }
    }
    if all.len() == front.len() {
        return front;
    }
    let objs: Vec<Vec<f64>> = all.iter().map(|p| evaluate(g, p, objset)).collect();
    let ranks = fast_nondominated_sort(&objs);
    all.into_iter()
        .zip(ranks)
        .filter(|(_, r)| *r == 1)
        .map(|(l, _)| l)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_pendant() -> CsrGraph {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
            (0, 6),
        ];
        CsrGraph::from_edges(&nodes, &edges)
    }

    fn unit_weights(g: &CsrGraph) -> Vec<f64> {
        vec![1.0; g.adj.len()]
    }

    #[test]
    fn refine_front_absorbs_singleton_and_is_at_least_as_good() {
        let g = graph_with_pendant();
        let w = unit_weights(&g);
        let part: Labels = vec![0, 0, 0, 3, 3, 3, 6];
        let front = vec![part.clone()];
        let refined = refine_front(&g, &w, front, ObjSet::default());
        assert!(!refined.is_empty());
        let absorbed = refined.iter().any(|p| p[6] == p[0]);
        assert!(
            absorbed,
            "refinement did not absorb the singleton: {refined:?}"
        );
        assert!(refined.iter().all(|p| p.len() == 7));
    }

    #[test]
    fn refine_front_empty_is_empty() {
        let g = graph_with_pendant();
        let w = unit_weights(&g);
        assert!(refine_front(&g, &w, Vec::new(), ObjSet::default()).is_empty());
    }

    #[test]
    fn refine_tiny_merge_target_follows_edge_weights() {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 6),
            (3, 6),
        ];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let part: Labels = vec![0, 0, 0, 3, 3, 3, 6];
        let weight_edge = |w: &mut [f64], a: usize, b: usize, val: f64| {
            for (u, v) in [(a, b), (b, a)] {
                let start = g.xadj[u] as usize;
                let end = g.xadj[u + 1] as usize;
                for p in start..end {
                    if g.adj[p] as usize == v {
                        w[p] = val;
                    }
                }
            }
        };
        let mut w = vec![1.0; g.adj.len()];
        weight_edge(&mut w, 3, 6, 5.0);
        assert_eq!(refine_tiny(&g, &w, &part, 2)[6], 3, "heavier side lost");
        let mut w = vec![1.0; g.adj.len()];
        weight_edge(&mut w, 2, 6, 5.0);
        assert_eq!(refine_tiny(&g, &w, &part, 2)[6], 0, "heavier side lost");
    }

    #[test]
    fn split_components_separates_disconnected_community() {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let split = split_components(&g, &vec![0; 6]).expect("no split produced");
        assert_eq!(split[0], split[1]);
        assert_eq!(split[1], split[2]);
        assert_eq!(split[3], split[4]);
        assert_eq!(split[4], split[5]);
        assert_ne!(split[0], split[3]);

        let connected: Labels = vec![0, 0, 0, 3, 3, 3];
        assert!(split_components(&g, &connected).is_none());
    }

    #[test]
    fn refine_front_split_copy_dominates_disconnected_member() {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = unit_weights(&g);
        let refined = refine_front(&g, &w, vec![vec![0; 6]], ObjSet::default());
        assert!(!refined.is_empty());
        assert!(
            refined.iter().all(|p| p[0] != p[3]),
            "disconnected member survived: {refined:?}"
        );
    }
}
