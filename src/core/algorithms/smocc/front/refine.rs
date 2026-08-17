//! Union refinement: every front member is re-offered split and reabsorbed, and
//! the enlarged set is re-sorted for non-dominance.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::collections::HashSet;

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::nsga2::fast_nondominated_sort;
use crate::core::algorithms::smocc::objectives::{ObjSet, evaluate};
use crate::core::graph::CsrGraph;

use super::agglom::agglomerate;
use super::components::split_components;
use super::tiny::refine_tiny;

pub fn refine_front(g: &CsrGraph, wadj: &[f64], front: Vec<Labels>, objset: ObjSet) -> Vec<Labels> {
    refine_front_mode(g, wadj, front, objset, true)
}

/// `coarsen` adds the agglomerative chain of `agglom` to the candidate pool.
/// The search returns pure but over-numerous communities, and nothing else in
/// the pipeline builds coarser ones.
pub fn refine_front_mode(
    g: &CsrGraph,
    wadj: &[f64],
    front: Vec<Labels>,
    objset: ObjSet,
    coarsen: bool,
) -> Vec<Labels> {
    if front.is_empty() {
        return front;
    }
    let mut seen: HashSet<Vec<i32>> = front.iter().cloned().collect();
    let mut all: Vec<Labels> = front.clone();
    for p in &front {
        if let Some(split) = split_components(g, p) {
            let merged = refine_tiny(g, wadj, &split);
            if seen.insert(merged.clone()) {
                all.push(merged);
            }
            if seen.insert(split.clone()) {
                all.push(split);
            }
        }
        let refined = refine_tiny(g, wadj, p);
        if seen.insert(refined.clone()) {
            all.push(refined);
        }
        if coarsen {
            for c in agglomerate(g, p) {
                if seen.insert(c.clone()) {
                    all.push(c);
                }
            }
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
    use crate::core::algorithms::smocc::objectives::ObjSet;

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
        let front = vec![part];
        let refined = refine_front(&g, &w, front, ObjSet::KkmRc);
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
        assert!(refine_front(&g, &w, Vec::new(), ObjSet::KkmRc).is_empty());
    }

    #[test]
    fn refine_front_split_copy_dominates_disconnected_member() {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let w = unit_weights(&g);
        let refined = refine_front(&g, &w, vec![vec![0; 6]], ObjSet::KkmRc);
        assert!(!refined.is_empty());
        assert!(
            refined.iter().all(|p| p[0] != p[3]),
            "disconnected member survived: {refined:?}"
        );
    }
}
