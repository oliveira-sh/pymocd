//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::rimpso::Labels;
use crate::core::graph::CsrGraph;

/// Per community: internal edges and degree sum, over the non-isolated vertices.
///
/// An isolated vertex is a member of nothing — the flight never places one — so
/// counting it would put a community in the model that the data never spoke to.
fn load(g: &CsrGraph, p: &Labels, e_c: &mut [f64], d_c: &mut [f64], live: &mut Vec<u32>) {
    for &c in live.iter() {
        e_c[c as usize] = 0.0;
        d_c[c as usize] = 0.0;
    }
    live.clear();
    for (u, &label) in p.iter().enumerate().take(g.n) {
        let deg = g.deg[u];
        if deg == 0 {
            continue;
        }
        let c = label as usize;
        if d_c[c] == 0.0 {
            live.push(c as u32);
        }
        d_c[c] += f64::from(deg);
    }
    for &(u, v) in &g.edges {
        let (a, b) = (p[u as usize], p[v as usize]);
        if a == b {
            e_c[a as usize] += 1.0;
        }
    }
}

/// How well the partition fits, as a degree-corrected assortative block model,
/// against how much of that fit its own free parameters could have bought.
///
/// The model gives every community its own internal edge density and lets
/// everything between communities share one:
///
/// ```text
/// A_ij ~ Poisson(k_i k_j omega_c)   inside community c
/// A_ij ~ Poisson(k_i k_j omega_out) between communities
/// ```
///
/// Profiling out the densities leaves a log-likelihood ratio against the
/// configuration model that is a sum of two bincounts:
///
/// ```text
/// L(C) = sum_c e_c ln(e_c / E_c)  +  e_out ln(e_out / E_out)
/// E_c  = d_c^2 / 4m,   E_out = m - sum_c E_c
/// ```
///
/// `E_c` is the number of intra-community edges the configuration model expects,
/// so `L` is exactly how many nats of edge placement the partition explains that
/// the degrees alone do not. Each community brings a free density, so the
/// densities are charged the Schwarz penalty: half a nat per parameter per log
/// observation, over the `B + 1` densities and the graph's `2m` edge endpoints.
/// `B` counts the communities holding at least one non-isolated vertex, because
/// `load` skips degree-zero vertices entirely.
///
/// ```text
/// score(C) = L(C) - (B + 1) / 2 * ln(2m)
/// ```
///
/// Note that `L` does NOT rise monotonically with `B`. That holds for the
/// unrestricted block model, where a refinement's model nests the coarser one;
/// here it does not, because splitting a community moves the pairs it sheds off
/// its own free density and onto the single shared `omega_out`. The test
/// `splitting_a_clique_is_not_worth_the_extra_density` is the case: the
/// refinement's `L` is lower than the planted partition's before any penalty.
///
/// Both degenerate partitions lose, neither by a special case. One community
/// gives `e_1 = m` and `E_1 = m`, so `L = 0` and the score is `-ln(2m)`. All
/// singletons give `e_c = 0` everywhere, so `edges_in = 0` while `expect_in > 0`,
/// and the assortativity guard below rejects the partition outright. There is no
/// degeneracy filter, no fallback stage and nothing to abstain with.
fn fit(g: &CsrGraph, p: &Labels, e_c: &mut [f64], d_c: &mut [f64], live: &mut Vec<u32>) -> f64 {
    if g.m == 0 {
        return 0.0;
    }
    load(g, p, e_c, d_c, live);
    let m = g.m as f64;

    let mut evidence = 0.0;
    let mut expect_in = 0.0;
    let mut edges_in = 0.0;
    for &c in live.iter() {
        let c = c as usize;
        let expect = d_c[c] * d_c[c] / (4.0 * m);
        expect_in += expect;
        edges_in += e_c[c];
        if e_c[c] > 0.0 && expect > 0.0 {
            evidence += e_c[c] * (e_c[c] / expect).ln();
        }
    }
    let (edges_out, expect_out) = (m - edges_in, m - expect_in);
    // Assortativity, made explicit. A partition whose communities are internally
    // SPARSER than the space between them fits the model just as well with the
    // two densities swapped, and it is not a community structure. No such
    // partition can sit on a CPM front, but `rimpso_select` takes candidates
    // from anywhere, so the constraint is stated rather than assumed.
    if edges_out > 0.0 {
        if edges_in * expect_out <= edges_out * expect_in {
            return f64::NEG_INFINITY;
        }
        if expect_out > 0.0 {
            evidence += edges_out * (edges_out / expect_out).ln();
        }
    }
    evidence - 0.5 * (live.len() as f64 + 1.0) * (2.0 * m).ln()
}

/// The archive member the degree-corrected assortative block model fits best,
/// once its own free densities are paid for. One criterion, no stages, no
/// fallback, no abstention.
pub fn select_index(g: &CsrGraph, front: &[Labels]) -> usize {
    front
        .par_iter()
        .enumerate()
        .map_init(
            || (vec![0.0f64; g.n], vec![0.0f64; g.n], Vec::new()),
            |(e_c, d_c, live), (i, p)| (fit(g, p, e_c, d_c, live), i),
        )
        .reduce_with(|a, b| if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) { b } else { a })
        .map_or(0, |(_, i)| i)
}

pub fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
    let pick = select_index(g, &front);
    front.into_iter().nth(pick).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::rimpso::utils::fixtures::ring_of_cliques;

    fn score(g: &CsrGraph, p: &Labels) -> f64 {
        let (mut e_c, mut d_c, mut live) = (vec![0.0; g.n], vec![0.0; g.n], Vec::new());
        fit(g, p, &mut e_c, &mut d_c, &mut live)
    }

    #[test]
    fn both_degenerate_partitions_score_below_the_planted_one() {
        let g = ring_of_cliques(6, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let best = score(&g, &planted);
        assert!(score(&g, &vec![0; g.n]) < best);
        assert!(score(&g, &(0..g.n as i32).collect::<Labels>()) < best);
        assert!(score(&g, &vec![0; g.n]) < 0.0);
    }

    #[test]
    fn one_community_has_no_evidence_at_all() {
        let g = ring_of_cliques(6, 5);
        let m = g.m as f64;
        assert!((score(&g, &vec![0; g.n]) + (2.0 * m).ln()).abs() < 1e-9);
    }

    #[test]
    fn the_planted_cliques_beat_every_coarsening_of_them() {
        let g = ring_of_cliques(12, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let best = score(&g, &planted);
        for group in [2usize, 3, 4, 6, 12] {
            let coarse: Labels = (0..g.n)
                .map(|i| ((i / (5 * group)) * 5 * group) as i32)
                .collect();
            assert!(best > score(&g, &coarse), "lost to {group} cliques per module");
        }
    }

    #[test]
    fn splitting_a_clique_is_not_worth_the_extra_density() {
        let g = ring_of_cliques(12, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let split: Labels = (0..g.n)
            .map(|i| if i % 5 < 3 { (i as i32 / 5) * 5 } else { i as i32 })
            .collect();
        assert!(score(&g, &planted) > score(&g, &split));
    }

    #[test]
    fn a_partition_sparser_than_chance_never_wins() {
        // Complete bipartite: splitting it by side puts no edge inside a community.
        let mut e = Vec::new();
        for a in 0..6i32 {
            for b in 6..12i32 {
                e.push((a, b));
            }
        }
        let g = CsrGraph::from_edges(&(0..12).collect::<Vec<i32>>(), &e);
        let sides: Labels = (0..12).map(|i| i32::from(i >= 6)).collect();
        assert_eq!(score(&g, &sides), f64::NEG_INFINITY);
        assert!(score(&g, &vec![0; g.n]).is_finite());
    }

    #[test]
    fn the_choice_is_the_same_however_the_archive_is_ordered() {
        let g = ring_of_cliques(10, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let front = vec![
            vec![0; g.n],
            (0..g.n).map(|i| (i as i32 / 10) * 10).collect(),
            planted,
            (0..g.n as i32).collect(),
        ];
        let a = front[select_index(&g, &front)].clone();
        let rev: Vec<Labels> = front.into_iter().rev().collect();
        assert_eq!(rev[select_index(&g, &rev)], a);
    }

    #[test]
    fn isolated_vertices_do_not_change_the_score() {
        let g = ring_of_cliques(6, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let base = score(&g, &planted);

        let mut nodes: Vec<i32> = (0..g.n as i32).collect();
        nodes.extend(g.n as i32..g.n as i32 + 20);
        let edges: Vec<(i32, i32)> = g.edges.iter().map(|&(u, v)| (u as i32, v as i32)).collect();
        let padded = CsrGraph::from_edges(&nodes, &edges);
        let mut p = planted.clone();
        p.resize(padded.n, 0);
        assert!((score(&padded, &p) - base).abs() < 1e-9);
    }

    #[test]
    fn the_buffers_are_reusable_across_members() {
        let g = ring_of_cliques(8, 5);
        let (mut e_c, mut d_c, mut live) = (vec![0.0; g.n], vec![0.0; g.n], Vec::new());
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let a = fit(&g, &planted, &mut e_c, &mut d_c, &mut live);
        fit(&g, &(0..g.n as i32).collect::<Labels>(), &mut e_c, &mut d_c, &mut live);
        let b = fit(&g, &planted, &mut e_c, &mut d_c, &mut live);
        assert_eq!(a, b, "the buffers leaked between members");
    }
}
