//! Choosing one partition out of the archive, without ground truth.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::config::defaults::DEFAULT_SELECT_MODE;
use crate::core::algorithms::mopso::objectives::{Obj, community_count};
use crate::core::graph::CsrGraph;

use super::modularity::max_modularity;
use super::plateau::widest_plateau;

const DEGENERATE_FRACTION: f64 = 0.5; // a member with at least this fraction of n communities is all singletons; never chosen.

/// The community count of every archive member.
pub fn counts(g: &CsrGraph, front: &[Labels]) -> Vec<usize> {
    front
        .par_iter()
        .map_init(
            || vec![false; g.n],
            |seen, p| community_count(p, seen),
        )
        .collect()
}

/// The resolution ladder's range in the selector's units: `gamma` over `[1/n^2, 1]`
/// divided by the graph's density.
fn lambda_span(g: &CsrGraph) -> (f64, f64) {
    let n = g.n as f64;
    let density = 2.0 * g.m as f64 / (n * (n - 1.0));
    if density <= 0.0 || density.is_nan() {
        return (0.0, f64::INFINITY);
    }
    (1.0 / (n * n * density), 1.0 / density)
}

/// The scalarised rule: the CPM optimum at `gamma` = the graph's own density, where the
/// two objectives carry equal weight.
fn cheapest(objs: &[Obj], keep: &[usize]) -> usize {
    let mut pick = keep[0];
    let mut best = objs[pick][0] + objs[pick][1];
    for &i in &keep[1..] {
        let s = objs[i][0] + objs[i][1];
        if s < best {
            best = s;
            pick = i;
        }
    }
    pick
}

pub fn select_index(g: &CsrGraph, front: &[Labels], objs: &[Obj]) -> usize {
    select_index_mode(g, front, objs, DEFAULT_SELECT_MODE)
}

/// Picks a member by `mode`: `1` the resolution plateau, falling back to modularity when
/// the hull is too thin to read; `2` modularity alone; anything else the equal-weight sum.
pub fn select_index_mode(g: &CsrGraph, front: &[Labels], objs: &[Obj], mode: u8) -> usize {
    if front.is_empty() {
        return 0;
    }
    let k = counts(g, front);
    let cap = (DEGENERATE_FRACTION * g.n as f64).max(2.0);
    let mut keep: Vec<usize> = (0..front.len())
        .filter(|&i| k[i] > 1 && (k[i] as f64) < cap)
        .collect();
    if keep.is_empty() {
        keep = (0..front.len()).collect();
    }

    match mode {
        1 => widest_plateau(objs, &k, &keep, lambda_span(g))
            .unwrap_or_else(|| max_modularity(g, front, objs, &keep)),
        2 => max_modularity(g, front, objs, &keep),
        _ => cheapest(objs, &keep),
    }
}

pub fn select_best(g: &CsrGraph, front: Vec<Labels>, objs: &[Obj]) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let pick = select_index(g, &front, objs);
    front.into_iter().nth(pick).unwrap()
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

    #[test]
    fn every_mode_rejects_the_degenerate_members() {
        let g = two_cliques();
        let split: Labels = (0..g.n).map(|i| i32::from(i >= 5) * 5).collect();
        let front = vec![vec![0; g.n], (0..g.n as i32).collect(), split.clone()];
        let objs = evaluated(&g, &front);
        for mode in [0u8, 1, 2] {
            let pick = select_index_mode(&g, &front, &objs, mode);
            assert_eq!(front[pick], split, "mode {mode} took a degenerate member");
        }
    }

    #[test]
    fn plateau_recovers_the_planted_ring_where_the_sum_does_not() {
        let (k, s) = (12i32, 5i32);
        let g = ring_of_cliques(k, s);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / s) * s).collect();
        let coarse: Labels = (0..g.n).map(|i| (i as i32 / (2 * s)) * 2 * s).collect();
        let one: Labels = vec![0; g.n];
        let fine: Labels = (0..g.n as i32).collect();
        let front = vec![one, coarse, planted.clone(), fine];
        let objs = evaluated(&g, &front);

        let pick = select_index_mode(&g, &front, &objs, 1);
        assert_eq!(front[pick], planted, "the plateau missed the planted ring");
    }

    #[test]
    fn selection_does_not_depend_on_archive_order() {
        let g = ring_of_cliques(8, 6);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 6) * 6).collect();
        let coarse: Labels = (0..g.n).map(|i| (i as i32 / 12) * 12).collect();
        let front = vec![vec![0; g.n], coarse, planted, (0..g.n as i32).collect()];
        let objs = evaluated(&g, &front);
        let a = front[select_index(&g, &front, &objs)].clone();

        let rev: Vec<Labels> = front.into_iter().rev().collect();
        let objs = evaluated(&g, &rev);
        assert_eq!(rev[select_index(&g, &rev, &objs)], a);
    }

    #[test]
    fn a_narrow_plateau_hands_over_to_modularity() {
        // The archive holds nothing between the planted partition and one community, so
        // the plateau the coarse member reads is inherited rather than earned.
        let g = ring_of_cliques(20, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let coarse: Labels = (0..g.n).map(|i| (i as i32 / 50) * 50).collect();
        let front = vec![planted.clone(), coarse, vec![0; g.n]];
        let objs = evaluated(&g, &front);
        assert_eq!(
            front[select_index_mode(&g, &front, &objs, 1)],
            planted,
            "the gate let a two-member hull decide"
        );
    }

    #[test]
    fn a_single_member_front_returns_it() {
        let g = two_cliques();
        let front = vec![vec![0; g.n]];
        let objs = evaluated(&g, &front);
        assert_eq!(select_index(&g, &front, &objs), 0);
        assert_eq!(select_best(&g, front, &objs), vec![0; g.n]);
    }
}
