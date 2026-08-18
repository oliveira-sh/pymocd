//! Resolution-plateau selection: keep the granularity surviving the longest sweep of `gamma`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopso::objectives::Obj;

/// A vertex of the lower-left convex hull, carrying the index of the front member it came
/// from and that member's community count.
struct Vertex {
    idx: usize,
    pair: f64,
    cut: f64,
    k: usize,
}

const ANCHOR: usize = usize::MAX; // marks the two anchors, which bound the profile but are never chosen.

/// The lower-left convex hull of the candidates, ordered by `pair` ascending; only its
/// vertices optimise `cut + lambda * pair` for some `lambda > 0`. The two degenerate
/// partitions join as anchors, so the coarsest and finest real members still get an
/// interval instead of landing on an endpoint.
fn lower_hull(objs: &[Obj], counts: &[usize], keep: &[usize]) -> Vec<Vertex> {
    let mut pts: Vec<Vertex> = keep
        .iter()
        .map(|&i| Vertex {
            idx: i,
            pair: objs[i][1],
            cut: objs[i][0],
            k: counts[i],
        })
        .chain([
            Vertex {
                idx: ANCHOR,
                pair: 0.0,
                cut: 1.0,
                k: 0,
            },
            Vertex {
                idx: ANCHOR,
                pair: 1.0,
                cut: 0.0,
                k: 0,
            },
        ])
        .collect();
    // Index last, so a tie in both objectives resolves the same way every run.
    pts.sort_unstable_by(|a, b| {
        a.pair
            .partial_cmp(&b.pair)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.cut
                    .partial_cmp(&b.cut)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.idx.cmp(&b.idx))
    });

    let mut hull: Vec<Vertex> = Vec::with_capacity(pts.len());
    let mut best_cut = f64::INFINITY;
    for p in pts {
        // The staircase first: at equal or greater `pair`, a member with no better `cut`
        // is dominated. The anchors are exempt, or a disconnected graph would delete the
        // one-community anchor with its own `cut = 0` partition into components.
        if p.idx != ANCHOR && p.cut >= best_cut {
            continue;
        }
        best_cut = best_cut.min(p.cut);
        while hull.len() >= 2 {
            let a = &hull[hull.len() - 2];
            let b = &hull[hull.len() - 1];
            let turn = (b.cut - a.cut) * (p.pair - a.pair) - (p.cut - a.cut) * (b.pair - a.pair);
            if turn >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

/// The scale two community counts share: a partition with twice as many communities is a
/// different answer, one with a few more is the same answer resampled.
fn octave(k: usize) -> usize {
    k.max(1).ilog2() as usize
}

const MIN_HULL: usize = 5; // fewest hull vertices for a plateau to be evidence; two are always anchors.

/// The index of the front member whose community count owns the widest span of resolutions.
///
/// Each interior hull vertex is the CPM optimum for `lambda` between the slopes of its two
/// incident hull edges, and the width of that interval in log `lambda` is how long the
/// partition survives as the resolution sweeps. Widths are pooled by octave of the
/// community count, because a dense front subdivides one true plateau across several
/// nearly collinear vertices and no single one keeps the full width.
///
/// A vertex whose interval runs past either end of `span` is dropped rather than
/// truncated: a width the range decided is not evidence about the graph.
///
/// Returns `None` when the hull is too thin, leaving the caller on its fallback.
pub fn widest_plateau(
    objs: &[Obj],
    counts: &[usize],
    keep: &[usize],
    span: (f64, f64),
) -> Option<usize> {
    let hull = lower_hull(objs, counts, keep);
    if hull.len() < MIN_HULL {
        return None;
    }
    let (lam_lo, lam_hi) = span;

    let mut spans: Vec<(usize, f64, f64, usize)> = Vec::new(); // octave, its total log-width, its best single width, that member.
    for i in 1..hull.len() - 1 {
        if hull[i].idx == ANCHOR {
            continue;
        }
        let hi = -(hull[i].cut - hull[i - 1].cut) / (hull[i].pair - hull[i - 1].pair);
        let lo = -(hull[i + 1].cut - hull[i].cut) / (hull[i + 1].pair - hull[i].pair);
        if !(lo >= lam_lo && hi <= lam_hi && hi > lo) {
            continue;
        }
        let w = hi.ln() - lo.ln();
        let k = octave(hull[i].k);
        match spans.iter_mut().find(|s| s.0 == k) {
            Some(s) => {
                s.1 += w;
                if w > s.2 {
                    s.2 = w;
                    s.3 = hull[i].idx;
                }
            }
            None => spans.push((k, w, w, hull[i].idx)),
        }
    }

    spans
        .into_iter()
        .reduce(|a, b| if b.1 > a.1 { b } else { a })
        .map(|s| s.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDE: (f64, f64) = (1e-12, 1e12); // wide enough to clip nothing, so these test the rule not the clip.

    #[test]
    fn a_wide_plateau_beats_a_crowded_shoulder() {
        // Three shoulder points packed together at fine granularity against one lone
        // vertex holding a wide slope interval.
        let objs = vec![
            [1.00, 0.000], // all singletons
            [0.90, 0.010],
            [0.88, 0.012],
            [0.86, 0.014],
            [0.84, 0.017],
            [0.82, 0.021],
            [0.40, 0.200], // the plateau
            [0.00, 1.000], // one community
        ];
        let counts = vec![100, 60, 55, 50, 45, 40, 8, 1];
        let keep: Vec<usize> = (0..objs.len()).collect();
        assert_eq!(widest_plateau(&objs, &counts, &keep, WIDE), Some(6));
    }

    #[test]
    fn octaves_group_by_doubling() {
        assert_eq!(octave(1), 0);
        assert_eq!(octave(2), 1);
        assert_eq!(octave(3), 1);
        assert_eq!(octave(4), 2);
        assert_eq!(
            octave(1092),
            octave(1313),
            "one plateau split across octaves"
        );
        assert_ne!(
            octave(1092),
            octave(2500),
            "a doubling stayed in one octave"
        );
        assert_eq!(octave(0), 0, "a zero count must not panic");
    }

    #[test]
    fn counts_are_pooled_across_collinear_vertices() {
        // The same granularity split over three nearly collinear hull vertices must still
        // out-span a single vertex of another granularity.
        let objs = vec![
            [1.00, 0.000],
            [0.90, 0.004],
            [0.80, 0.011],
            [0.62, 0.050],
            [0.60, 0.056],
            [0.58, 0.063],
            [0.20, 0.400],
            [0.00, 1.000],
        ];
        let counts = vec![600, 140, 33, 9, 10, 11, 3, 1]; // the three middle members are one granularity resampled.
        let keep: Vec<usize> = (0..objs.len()).collect();
        let pick = widest_plateau(&objs, &counts, &keep, WIDE).expect("no plateau found");
        assert!(
            (9..=11).contains(&counts[pick]),
            "the pooled octave lost: picked k={}",
            counts[pick]
        );
    }

    #[test]
    fn interior_points_off_the_hull_never_win() {
        let objs = vec![
            [1.00, 0.000],
            [0.90, 0.004],
            [0.80, 0.011],
            [0.50, 0.100],
            [0.99, 0.990], // strictly inside, and dominated
            [0.00, 1.000],
        ];
        let counts = vec![400, 90, 20, 5, 4, 1]; // octave-separated, so the pooling cannot decide this one.
        let keep: Vec<usize> = (0..objs.len()).collect();
        assert_eq!(widest_plateau(&objs, &counts, &keep, WIDE), Some(3));
    }

    #[test]
    fn a_hull_too_thin_to_be_evidence_declines() {
        let objs = vec![[1.0, 0.0], [0.4, 0.2], [0.0, 1.0]];
        let counts = vec![9, 3, 1]; // three vertices leave one interior candidate, which would win by default.
        assert!(widest_plateau(&objs, &counts, &[0, 1, 2], WIDE).is_none());
        assert!(widest_plateau(&objs, &counts, &[0, 1], WIDE).is_none());
        assert!(widest_plateau(&objs, &counts, &[0], WIDE).is_none());
    }

    #[test]
    fn the_pick_is_independent_of_input_order() {
        let objs = vec![
            [1.00, 0.000],
            [0.90, 0.010],
            [0.70, 0.030],
            [0.55, 0.070],
            [0.40, 0.200],
            [0.00, 1.000],
        ];
        let counts = vec![100, 60, 30, 15, 8, 1];
        let fwd: Vec<usize> = (0..6).collect();
        let rev: Vec<usize> = (0..6).rev().collect();
        assert_eq!(
            widest_plateau(&objs, &counts, &fwd, WIDE),
            widest_plateau(&objs, &counts, &rev, WIDE)
        );
    }
}
