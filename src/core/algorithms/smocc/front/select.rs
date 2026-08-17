//! The label-free selector: min-max normalise all four objectives over the
//! front and keep the member of least total normalised cost.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::objectives;
use crate::core::graph::CsrGraph;

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

    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
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

    let score = |o: &[f64; 4]| -> f64 {
        let mut s = 0.0;
        for c in 0..4 {
            let rng = hi[c] - lo[c];
            if rng > 0.0 {
                s += (o[c] - lo[c]) / rng;
            }
        }
        s
    };

    let mut pick = 0usize;
    let mut best = score(&obj[0]);
    for (j, o) in obj.iter().enumerate().skip(1) {
        let s = score(o);
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
