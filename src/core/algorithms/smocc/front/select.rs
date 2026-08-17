//! The label-free selector: min-max normalise all four objectives over the
//! front and keep the member of least total normalised cost.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use rustc_hash::FxHashSet;

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::config::defaults::DEFAULT_SELECT_MODE;
use crate::core::algorithms::smocc::objectives;
use crate::core::graph::CsrGraph;

fn community_count(p: &Labels) -> usize {
    p.iter().collect::<FxHashSet<_>>().len()
}

pub fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let pick = select_index(g, &front);
    front.into_iter().nth(pick).unwrap()
}

/// The index `select_best` picks, so callers can compare selection rules over
/// an identical candidate set without re-running the search.
pub fn select_index(g: &CsrGraph, front: &[Labels]) -> usize {
    select_index_mode(g, front, DEFAULT_SELECT_MODE)
}

/// Percentile of the retained scores that anchors each objective's scale in
/// the robust mode, and its complement at the top.
const ROBUST_LO: f64 = 0.05;

/// A community count at or above this fraction of `n` marks a degenerate
/// member: an all-singletons partition, or nearly one.
const DEGENERATE_FRACTION: f64 = 0.5;

/// Fewest non-degenerate members for the percentile anchors to mean anything.
const MIN_SCALE: usize = 16;

/// Fewest vertices for the correction to apply. The distortion it removes is
/// that a front spanning orders of magnitude in community count lets its
/// extremes set every objective's scale; on a graph of a few dozen vertices
/// the front spans no such range and the plain min-max is the better rule.
/// The paper reports the measurement behind this threshold.
const MIN_VERTICES: usize = 500;

/// `mode` selects the normalisation.
///
/// `0` is min-max over the whole front. The front spans everything from one
/// community to all singletons, so its extremes set the scale for every
/// objective and compress the candidates a user would actually consider into a
/// narrow band; the sum then lands finer than the graph warrants.
///
/// `1` first drops the degenerate members, then anchors each objective's scale
/// at the 5th and 95th percentiles of what remains, so a handful of outlying
/// members cannot set the scale.
pub fn select_index_mode(g: &CsrGraph, front: &[Labels], mode: u8) -> usize {
    if front.is_empty() {
        return 0;
    }
    let obj: Vec<[f64; 4]> = front
        .par_iter()
        .map(|p| {
            let (kkm, rc) = objectives::kkm_rc(g, p);
            let (intra, inter) = objectives::intra_inter(g, p);
            [kkm, rc, intra, inter]
        })
        .collect();

    // Which members set the scale and may be chosen. In the robust mode the
    // degenerate ones do neither: scoring them against a scale they did not set
    // lets a partition clamp to zero on the objectives it minimises and win on
    // that alone, and a single community or all singletons is never the answer
    // a caller wants.
    let mut scale: Vec<usize> = (0..front.len()).collect();
    let mut robust = false;
    if mode == 1 {
        let cap = (DEGENERATE_FRACTION * g.n as f64).max(2.0);
        let kept: Vec<usize> = scale
            .iter()
            .copied()
            .filter(|&i| {
                let k = community_count(&front[i]);
                k > 1 && (k as f64) < cap
            })
            .collect();
        if kept.len() >= MIN_SCALE && g.n >= MIN_VERTICES {
            scale = kept;
            robust = true;
        }
    }

    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
    if robust {
        let mut col: Vec<f64> = Vec::with_capacity(scale.len());
        for c in 0..4 {
            col.clear();
            col.extend(scale.iter().map(|&i| obj[i][c]));
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let last = col.len() - 1;
            let at = |q: f64| col[((last as f64) * q).round() as usize];
            lo[c] = at(ROBUST_LO);
            hi[c] = at(1.0 - ROBUST_LO);
        }
    } else {
        for o in &obj {
            for c in 0..4 {
                if o[c] < lo[c] {
                    lo[c] = o[c];
                }
                if o[c] > hi[c] {
                    hi[c] = o[c];
                }
            }
        }
    }

    // Values outside the anchored range clamp to it, so a member the scale
    // ignores can still be scored and can still win.
    let score = |o: &[f64; 4]| -> f64 {
        let mut s = 0.0;
        for c in 0..4 {
            let rng = hi[c] - lo[c];
            if rng > 0.0 {
                s += ((o[c] - lo[c]) / rng).clamp(0.0, 1.0);
            }
        }
        s
    };

    let mut pick = scale[0];
    let mut best = score(&obj[pick]);
    for &j in &scale[1..] {
        let s = score(&obj[j]);
        if s < best {
            best = s;
            pick = j;
        }
    }
    pick
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::utils::fixtures::two_clique_edges;

    #[test]
    fn selector_normalises_and_breaks_ties_by_lowest_index() {
        let nodes: Vec<i32> = (0..10).collect();
        let g = CsrGraph::from_edges(&nodes, &two_clique_edges());

        let split: Labels = (0..g.n).map(|i| i32::from(i >= 5)).collect();
        let one: Labels = vec![0; g.n];
        let sing: Labels = (0..g.n as i32).collect();
        assert_eq!(
            select_best(&g, vec![split.clone(), split.clone()]),
            split,
            "a constant column must contribute 0, not NaN"
        );
        assert_eq!(select_best(&g, vec![one.clone()]), one);

        let picked = select_best(&g, vec![one, sing, split.clone()]);
        assert_eq!(picked, split, "normalised scalarisation missed the split");
    }
}
