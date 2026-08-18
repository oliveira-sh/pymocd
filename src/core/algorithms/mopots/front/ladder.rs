//! The CPM resolution ladder: the lower convex hull of the front in (pair, intra) space.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopots::nsga2::Individual;

#[derive(Clone, Copy)]
struct Point {
    idx: usize,
    co_pairs: i128,
    intra_edges: i128,
}

/// The ladder of resolutions: `(index into front, gamma)` for every member that is the CPM
/// optimum over some `gamma > 0`, sorted by increasing gamma.
/// `front` members carry `objectives = [cut, pair]`, whose integer numerators are recovered
/// from the edge count `m` and `all_pairs = C(n_a,2)`; concave-dent members are omitted.
pub fn resolution_ladder(front: &[Individual], m: f64, all_pairs: f64) -> Vec<(usize, f64)> {
    if front.is_empty() {
        return Vec::new();
    }

    // cut and pair are integers over m and all_pairs, both under 2^53, so this rounds back exact
    let mut pts: Vec<Point> = front
        .iter()
        .enumerate()
        .map(|(idx, ind)| Point {
            idx,
            co_pairs: numerator(ind.objectives[1], all_pairs),
            intra_edges: numerator(1.0 - ind.objectives[0], m),
        })
        .collect();
    // pair ascending is co_pairs ascending, and cut ascending on a tie is intra descending
    pts.sort_by(|a, b| {
        a.co_pairs
            .cmp(&b.co_pairs)
            .then(b.intra_edges.cmp(&a.intra_edges))
    });
    // a rank-1 front holds no two equal-pair members, so this only guards the slices tests pass
    pts.dedup_by(|a, b| a.co_pairs == b.co_pairs);

    let hull = lower_convex_hull(pts);

    // gamma rises as the optimum gets finer, so the ladder walks the hull from the largest pair
    let last: usize = hull.len() - 1;
    let mut ladder: Vec<(usize, f64)> = Vec::with_capacity(hull.len());
    ladder.push((hull[last].idx, 0.0));

    for k in (0..last).rev() {
        let hi: Point = hull[k + 1];
        let lo: Point = hull[k];
        // gamma_d cancels against m and all_pairs, leaving a pure ratio of integer differences
        let d_intra: f64 = (lo.intra_edges - hi.intra_edges) as f64;
        let d_co: f64 = (lo.co_pairs - hi.co_pairs) as f64;
        let gamma: f64 = d_intra / d_co;
        if gamma.is_finite() && gamma > 0.0 {
            ladder.push((lo.idx, gamma));
        }
    }

    ladder
}

// Andrew's monotone chain over the pair-sorted points
fn lower_convex_hull(pts: Vec<Point>) -> Vec<Point> {
    let mut hull: Vec<Point> = Vec::with_capacity(pts.len());
    for p in pts {
        // pop while the middle point sits on or above the segment joining its neighbours
        while hull.len() >= 2 {
            let a: Point = hull[hull.len() - 2];
            let b: Point = hull[hull.len() - 1];
            // the turn is exact in i128, so a genuine vertex survives at every graph size
            if turn(a, b, p) > 0 {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

#[inline]
fn numerator(ratio: f64, denominator: f64) -> i128 {
    (ratio * denominator).round() as i128
}

// exact sign of z = (b − a) × (p − a) in (pair, cut) space, rescaled by the positive m·all_pairs
#[inline]
const fn turn(a: Point, b: Point, p: Point) -> i128 {
    (b.intra_edges - a.intra_edges) * (p.co_pairs - a.co_pairs)
        - (b.co_pairs - a.co_pairs) * (p.intra_edges - a.intra_edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopots::objectives::{Metrics, cpm};
    use crate::core::graph::Partition;

    fn member(cut: f64, pair: f64) -> Individual {
        let mut ind = Individual::new(Partition::default());
        ind.objectives = vec![cut, pair];
        ind
    }

    // the same member written the way the objectives build it, from its integer numerators
    fn counted(intra: f64, co: f64, m: f64, all_pairs: f64) -> Individual {
        member(1.0 - intra / m, co / all_pairs)
    }

    fn metrics(ind: &Individual) -> Metrics {
        Metrics {
            cut: ind.objectives[0],
            pair: ind.objectives[1],
        }
    }

    fn split(ladder: &[(usize, f64)]) -> (Vec<usize>, Vec<f64>) {
        (
            ladder.iter().map(|&(i, _)| i).collect(),
            ladder.iter().map(|&(_, g)| g).collect(),
        )
    }

    #[test]
    fn empty_front_has_no_ladder() {
        assert!(resolution_ladder(&[], 10.0, 10.0).is_empty());
    }

    #[test]
    fn single_member_front_is_the_whole_ladder() {
        // intra = 7 of m = 10, co = 4 of all_pairs = 10
        let front = vec![member(0.3, 0.4)];
        assert_eq!(resolution_ladder(&front, 10.0, 10.0), vec![(0, 0.0)]);
    }

    #[test]
    fn convex_three_point_front_keeps_every_member() {
        // (cut, pair): all singletons, a middle split, one giant community
        let front = vec![member(1.0, 0.0), member(0.4, 0.2), member(0.0, 1.0)];
        let ladder = resolution_ladder(&front, 10.0, 10.0);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![2, 1, 0], "coarsest first, finest last");
        assert_eq!(gammas[0], 0.0, "the coarsest member wins at gamma = 0");
        assert_eq!(gammas[1], 0.5, "(6-10)/(2-10)");
        assert_eq!(gammas[2], 3.0, "(0-6)/(0-2)");
        assert!(ladder.windows(2).all(|w| w[0].1 < w[1].1), "gammas rise");
    }

    #[test]
    fn concave_member_is_omitted() {
        // index 2 sits above the segment joining index 1 and index 3, so no gamma ever picks it
        let front = vec![
            member(1.0, 0.0),
            member(0.4, 0.2),
            member(0.35, 0.6),
            member(0.0, 1.0),
        ];
        let ladder = resolution_ladder(&front, 20.0, 20.0);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![3, 1, 0], "the dent is dropped, not reported");
        assert_eq!(gammas[1], 0.5, "(12-20)/(4-20)");
        assert_eq!(gammas[2], 3.0, "(0-12)/(0-4)");
        assert!(ladder.windows(2).all(|w| w[0].1 < w[1].1), "gammas rise");
    }

    #[test]
    fn the_denominators_set_the_scale_of_every_gamma() {
        // m = 10 gives intra 0, 6, 10 and all_pairs = 25 gives co 0, 5, 25
        let front = vec![member(1.0, 0.0), member(0.4, 0.2), member(0.0, 1.0)];
        let ladder = resolution_ladder(&front, 10.0, 25.0);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![2, 1, 0], "coarsest first, finest last");
        assert_eq!(gammas[0], 0.0, "the coarsest member wins at gamma = 0");
        assert_eq!(gammas[1], 0.2, "(6-10)/(5-25), not (6-10)/(2-10)");
        assert_eq!(gammas[2], 1.2, "(0-6)/(0-5), not (0-6)/(0-2)");
    }

    #[test]
    fn every_rung_maximises_cpm_strictly_inside_its_own_interval() {
        // m = 10 and all_pairs = 25, so the density cpm still needs is 10/25
        let (m, all_pairs) = (10.0, 25.0);
        let gamma_d = m / all_pairs;
        // index 2 is a concave dent, so no probe may ever hand it the maximum
        let front = vec![
            counted(0.0, 0.0, m, all_pairs),
            counted(6.0, 5.0, m, all_pairs),
            counted(7.0, 15.0, m, all_pairs),
            counted(10.0, 25.0, m, all_pairs),
        ];
        let ladder = resolution_ladder(&front, m, all_pairs);
        assert_eq!(ladder.len(), 3, "ladder = {ladder:?}");

        for (rung, &(winner, gamma)) in ladder.iter().enumerate() {
            // probe strictly inside the rung: midway to the next gamma, half again past the last
            let probe: f64 = match ladder.get(rung + 1) {
                Some(&(_, next)) => 0.5 * (gamma + next),
                None => gamma * 1.5,
            };
            let best: f64 = cpm(&metrics(&front[winner]), probe, gamma_d);
            for (i, other) in front.iter().enumerate() {
                let value: f64 = cpm(&metrics(other), probe, gamma_d);
                assert!(
                    best >= value - 1e-12,
                    "member {i} beats rung {rung} at gamma {probe}: {value} > {best}"
                );
            }
        }
    }

    #[test]
    fn a_member_exactly_on_a_hull_segment_is_omitted() {
        // exactly collinear on the two-triangles scales m = 7 and C(6,2) = 15
        let front = vec![
            member(1.0, 0.0),
            member(1.0 - 3.0 / 7.0, 3.0 / 15.0),
            member(1.0 - 6.0 / 7.0, 6.0 / 15.0),
        ];
        let ladder = resolution_ladder(&front, 7.0, 15.0);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![2, 0], "a collinear member owns no interval");
        assert_eq!(gammas[1], 1.0, "(0-6)/(0-6)");
    }

    #[test]
    fn an_edgeless_graph_yields_only_the_coarsest() {
        // m = 0 makes every intra 0, so every candidate gamma is 0 and none is a rung
        let front = vec![member(1.0, 0.0), member(0.4, 0.2), member(0.0, 1.0)];
        assert_eq!(resolution_ladder(&front, 0.0, 10.0), vec![(2, 0.0)]);
    }

    #[test]
    fn the_two_triangles_ladder_is_exact() {
        // two triangles joined by one bridge: m = 7, C(6,2) = 15
        let (m, all_pairs) = (7.0, 15.0);
        // singletons, one triangle split in two, one community per triangle, one blob
        let front = vec![
            counted(0.0, 0.0, m, all_pairs),
            counted(4.0, 4.0, m, all_pairs),
            counted(6.0, 6.0, m, all_pairs),
            counted(7.0, 15.0, m, all_pairs),
        ];
        let ladder = resolution_ladder(&front, m, all_pairs);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![3, 2, 0], "hull = (7,15), (6,6), (0,0)");
        assert_eq!(gammas[0], 0.0, "the blob wins at gamma = 0");
        // integer arithmetic hits 1/9 dead on where the f64 cross gave 0.11111111111111116
        assert_eq!(gammas[1], 1.0 / 9.0, "(6-7)/(6-15) = 1/9");
        assert_eq!(gammas[2], 1.0, "(0-6)/(0-6) = 1");
    }

    #[test]
    fn a_vertex_clearing_the_chord_by_one_pair_survives_at_scale() {
        // co 0, 999999999999, 1999999999999 against intra 0, 1, 2: the turn is exactly 1
        let (m, all_pairs) = (2.0, 2.0e12);
        let front = vec![
            counted(0.0, 0.0, m, all_pairs),
            counted(1.0, 999_999_999_999.0, m, all_pairs),
            counted(2.0, 1_999_999_999_999.0, m, all_pairs),
        ];
        let ladder = resolution_ladder(&front, m, all_pairs);
        let (idx, gammas) = split(&ladder);

        // the old 1e-12 relative tolerance measured this turn against ~4e12 and popped it
        assert_eq!(
            idx,
            vec![2, 1, 0],
            "one co-clustered pair is a genuine vertex"
        );
        assert_eq!(
            gammas[1],
            1.0 / 1_000_000_000_000.0,
            "(1-2)/(999999999999-1999999999999)"
        );
        assert_eq!(gammas[2], 1.0 / 999_999_999_999.0, "(0-1)/(0-999999999999)");
        assert!(ladder.windows(2).all(|w| w[0].1 < w[1].1), "gammas rise");
    }

    #[test]
    fn a_collinear_member_is_omitted_at_the_same_scale() {
        // the same shape one pair further along the chord: the turn is exactly 0, not 1
        let (m, all_pairs) = (2.0, 2.0e12);
        let front = vec![
            counted(0.0, 0.0, m, all_pairs),
            counted(1.0, 1_000_000_000_000.0, m, all_pairs),
            counted(2.0, 2_000_000_000_000.0, m, all_pairs),
        ];
        let ladder = resolution_ladder(&front, m, all_pairs);
        let (idx, gammas) = split(&ladder);

        assert_eq!(idx, vec![2, 0], "a collinear member owns no interval");
        assert_eq!(gammas[1], 2.0 / 2.0e12, "(0-2)/(0-2e12)");
    }
}
