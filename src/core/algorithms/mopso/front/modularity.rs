//! Newman modularity of an archive member, the selector's fallback ranking.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::objectives::Obj;
use crate::core::graph::CsrGraph;

/// Per-community degree mass, reused across the members one worker scores.
struct Mass {
    deg: Vec<f64>,
    live: Vec<u32>,
}

const UNSEEN: f64 = -1.0; // marks a community `deg` has not accumulated yet; zero cannot, an isolated vertex has it.

impl Mass {
    fn new(n: usize) -> Self {
        Self {
            deg: vec![UNSEEN; n],
            live: Vec::new(),
        }
    }

    /// `Q = (1 - cut) - sum_c (d_c / 2m)^2`. The first term is already in the archive, so
    /// scoring a member costs one pass over the labels and none over the edges.
    fn modularity(&mut self, g: &CsrGraph, labels: &[i32], cut: f64) -> f64 {
        for &c in &self.live {
            self.deg[c as usize] = UNSEEN;
        }
        self.live.clear();
        for (v, &c) in labels.iter().enumerate() {
            let c = c as usize;
            if self.deg[c] == UNSEEN {
                self.live.push(c as u32);
                self.deg[c] = 0.0;
            }
            self.deg[c] += f64::from(g.deg[v]);
        }
        let two_m = 2.0 * g.m as f64;
        let inter: f64 = self
            .live
            .iter()
            .map(|&c| (self.deg[c as usize] / two_m).powi(2))
            .sum();
        (1.0 - cut) - inter
    }
}

/// The candidate of greatest modularity, ties by the lower index.
pub fn max_modularity(g: &CsrGraph, front: &[Labels], objs: &[Obj], keep: &[usize]) -> usize {
    if g.m == 0 {
        return keep[0];
    }
    let scored: Vec<f64> = keep
        .par_iter()
        .map_init(
            || Mass::new(g.n),
            |mass, &i| mass.modularity(g, &front[i], objs[i][0]),
        )
        .collect();
    let mut pick = 0;
    for i in 1..keep.len() {
        if scored[i] > scored[pick] {
            pick = i;
        }
    }
    keep[pick]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::objectives::{measure, obj_of};
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_cliques};

    fn evaluated(g: &CsrGraph, front: &[Labels]) -> Vec<Obj> {
        let mut size = vec![0u32; g.n];
        let mut live = Vec::new();
        front
            .iter()
            .map(|p| obj_of(g, measure(g, p, &mut size, &mut live)))
            .collect()
    }

    /// Q straight from the definition.
    fn reference(g: &CsrGraph, labels: &[i32]) -> f64 {
        let m = g.m as f64;
        let mut q = 0.0;
        let mut seen: Vec<i32> = labels.to_vec();
        seen.sort_unstable();
        seen.dedup();
        for c in seen {
            let members: Vec<usize> = (0..g.n).filter(|&u| labels[u] == c).collect();
            let e: f64 = g
                .edges
                .iter()
                .filter(|&&(u, v)| labels[u as usize] == c && labels[v as usize] == c)
                .count() as f64;
            let d: f64 = members.iter().map(|&u| f64::from(g.deg[u])).sum();
            q += e / m - (d / (2.0 * m)).powi(2);
        }
        q
    }

    #[test]
    fn matches_the_definition() {
        let g = ring_of_cliques(6, 5);
        let mut mass = Mass::new(g.n);
        for part in [
            (0..g.n).map(|i| (i as i32 / 5) * 5).collect::<Labels>(),
            (0..g.n).map(|i| (i as i32 / 10) * 10).collect(),
            vec![0; g.n],
            (0..g.n as i32).collect(),
        ] {
            let objs = evaluated(&g, std::slice::from_ref(&part));
            let got = mass.modularity(&g, &part, objs[0][0]);
            assert!(
                (got - reference(&g, &part)).abs() < 1e-12,
                "{got} != {}",
                reference(&g, &part)
            );
        }
    }

    #[test]
    fn picks_the_planted_split_over_the_degenerates() {
        let g = two_cliques();
        let split: Labels = (0..g.n).map(|i| i32::from(i >= 5) * 5).collect();
        let front = vec![vec![0; g.n], (0..g.n as i32).collect(), split.clone()];
        let objs = evaluated(&g, &front);
        let keep: Vec<usize> = (0..front.len()).collect();
        assert_eq!(front[max_modularity(&g, &front, &objs, &keep)], split);
    }

    #[test]
    fn the_scratch_is_reusable_and_the_choice_is_order_free() {
        let g = ring_of_cliques(8, 5);
        let front: Vec<Labels> = vec![
            (0..g.n).map(|i| (i as i32 / 5) * 5).collect(),
            (0..g.n).map(|i| (i as i32 / 10) * 10).collect(),
            vec![0; g.n],
        ];
        let objs = evaluated(&g, &front);
        let a = front[max_modularity(&g, &front, &objs, &[0, 1, 2])].clone();
        let rev: Vec<Labels> = front.into_iter().rev().collect();
        let objs = evaluated(&g, &rev);
        assert_eq!(rev[max_modularity(&g, &rev, &objs, &[0, 1, 2])], a);
    }

    #[test]
    fn an_isolated_first_member_does_not_double_count() {
        // Vertex 0 is isolated and shares a community with a triangle, so the community's
        // degree mass is still zero when its second member arrives.
        let g = CsrGraph::from_edges(&[0, 1, 2, 3, 4], &[(1, 2), (2, 3), (1, 3)]);
        let part: Labels = vec![1, 1, 1, 1, 4];
        let objs = evaluated(&g, std::slice::from_ref(&part));
        let got = Mass::new(g.n).modularity(&g, &part, objs[0][0]);
        assert!(
            (got - reference(&g, &part)).abs() < 1e-12,
            "{got} != {}",
            reference(&g, &part)
        );
    }

    #[test]
    fn an_edgeless_graph_does_not_divide_by_zero() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        let front = vec![vec![0, 1, 2]];
        let objs = evaluated(&g, &front);
        assert_eq!(max_modularity(&g, &front, &objs, &[0]), 0);
    }
}
