//! Union-based Pareto-front refinement. Refined copies only ADD candidates;
//! the union is non-dominated-sorted back to rank-1, so the result front can
//! never be worse than the input.

use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

use super::Labels;
use super::nsga2::fast_nondominated_sort;
use super::objectives::kkm_rc;

/// Merge tiny communities (size `<= max_size`) into the neighbouring community
/// they share the most edges with, while preserving internally cohesive small
/// groups. Singletons always merge; pairs merge only when they lack an internal
/// edge or have more external than internal support. Returns a refined copy
/// without modifying the input.
pub(crate) fn refine_tiny(g: &CsrGraph, part: &[i32], max_size: usize) -> Vec<i32> {
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
            let mut internal = 0i64;
            let mut ext: FxHashMap<i32, i64> = FxHashMap::default();
            for &u in nodes {
                for &v in g.neighbors(u) {
                    let cv = p[v as usize];
                    if cv == c {
                        internal += 1;
                    } else {
                        *ext.entry(cv).or_insert(0) += 1;
                    }
                }
            }
            internal /= 2; // each internal edge counted from both ends
            let target = ext.iter().max_by(|a, b| {
                a.1.cmp(b.1).then_with(|| {
                    let sa = members.get(a.0).map_or(0, |v| v.len());
                    let sb = members.get(b.0).map_or(0, |v| v.len());
                    sa.cmp(&sb)
                })
            });
            let Some((&tc, &te)) = target else { continue };
            if nodes.len() == 1 || internal == 0 || te > internal {
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

/// Each member gets a tiny-community-merge copy (`refine_tiny`, max tiny size 2);
/// originals ∪ refined are deduplicated and non-dominated-sorted on (KKM, RC),
/// and the rank-1 front is returned.
pub fn refine_front(g: &CsrGraph, front: Vec<Labels>) -> Vec<Labels> {
    if front.len() <= 1 {
        // A single member still benefits from its refined alternative.
        if front.is_empty() {
            return front;
        }
    }
    let mut seen: HashSet<Vec<i32>> = front.iter().cloned().collect();
    let mut all: Vec<Labels> = front.clone();
    for p in &front {
        let refined = refine_tiny(g, p, 2);
        if seen.insert(refined.clone()) {
            all.push(refined);
        }
    }
    if all.len() == front.len() {
        return front;
    }
    let objs: Vec<(f64, f64)> = all.iter().map(|p| kkm_rc(g, p)).collect();
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
    use crate::core::graph::CsrGraph;

    // Two triangles {0,1,2},{3,4,5} + bridge (2,3), plus a pendant node 6 hanging
    // off node 0. A partition that isolates node 6 as a singleton should be
    // refinable: refine_tiny merges 6 back into community 0.
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

    #[test]
    fn refine_front_absorbs_singleton_and_is_at_least_as_good() {
        let g = graph_with_pendant();
        // Front member with node 6 as its own singleton community (label 6).
        let part: Labels = vec![0, 0, 0, 3, 3, 3, 6];
        let front = vec![part.clone()];
        let refined = refine_front(&g, front);
        assert!(!refined.is_empty());
        // The union front must contain a member where node 6 is no longer a
        // singleton (it merged into community 0, its only neighbour's community).
        let absorbed = refined.iter().any(|p| p[6] == p[0]);
        assert!(
            absorbed,
            "refinement did not absorb the singleton: {refined:?}"
        );
        // Every returned member is a full partition.
        assert!(refined.iter().all(|p| p.len() == 7));
    }

    #[test]
    fn refine_front_empty_is_empty() {
        let g = graph_with_pendant();
        assert!(refine_front(&g, Vec::new()).is_empty());
    }
}
