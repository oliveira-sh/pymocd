use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

use super::Labels;
use super::nsga2::fast_nondominated_sort;
use super::objectives::{ObjSet, evaluate};

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
            internal /= 2;
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

pub fn refine_front(g: &CsrGraph, front: Vec<Labels>, objset: ObjSet) -> Vec<Labels> {
    if front.is_empty() {
        return front;
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
    use crate::core::graph::CsrGraph;

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
        let part: Labels = vec![0, 0, 0, 3, 3, 3, 6];
        let front = vec![part.clone()];
        let refined = refine_front(&g, front, ObjSet::default());
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
        assert!(refine_front(&g, Vec::new(), ObjSet::default()).is_empty());
    }
}
