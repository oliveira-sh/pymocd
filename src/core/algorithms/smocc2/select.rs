//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::algorithms::smocc2::Labels;
use crate::core::algorithms::smocc2::objectives;
use crate::core::graph::CsrGraph;

pub(crate) fn select_best(g: &CsrGraph, front: Vec<Labels>) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let obj: Vec<[f64; 4]> = front
        .par_iter()
        .map(|p| {
            let (kkm, rc) = objectives::kkm_rc(g, p);
            let (intra, inter) = objectives::intra_inter(g, p);
            [kkm, rc, intra, inter]
        })
        .collect();
    pick_best(front, obj)
}

pub(crate) fn select_best_gpu(
    g: &CsrGraph,
    front: Vec<Labels>,
    dev: &mut crate::core::algorithms::smocc2::gpu::Gpu,
    cap: usize,
) -> Labels {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let mut obj: Vec<[f64; 4]> = Vec::with_capacity(front.len());
    for chunk in front.chunks(cap.max(1)) {
        let refs: Vec<&Labels> = chunk.iter().collect();
        obj.extend(
            dev.eval_labels(&refs)
                .expect("CUDA runtime failure in selection eval"),
        );
    }
    pick_best(front, obj)
}

fn pick_best(front: Vec<Labels>, obj: Vec<[f64; 4]>) -> Labels {

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
    front.into_iter().nth(pick).unwrap()
}


pub(crate) fn to_output(g: &CsrGraph, labels: &Labels) -> Vec<(i32, i32)> {
    let mut remap: FxHashMap<i32, i32> = FxHashMap::default();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(g.n);
    for (i, &d) in g.deg.iter().enumerate() {
        let comm = if d == 0 {
            -1
        } else {
            *remap.entry(labels[i]).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        };
        out.push((g.labels[i], comm));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_clique_edges() -> Vec<(i32, i32)> {
        let mut e = Vec::new();
        for (lo, hi) in [(0, 5), (5, 10)] {
            for a in lo..hi {
                for b in (a + 1)..hi {
                    e.push((a, b));
                }
            }
        }
        e.push((4, 5));
        e
    }

    #[test]
    fn selector_normalises_and_breaks_ties_by_lowest_index() {
        let nodes: Vec<i32> = (0..10).collect();
        let g = CsrGraph::from_edges(&nodes, &two_clique_edges());

        let split: Labels = (0..g.n).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let one: Labels = vec![0; g.n];
        let sing: Labels = (0..g.n as i32).collect();
        assert_eq!(
            select_best(&g, vec![split.clone(), split.clone()]),
            split,
            "a constant column must contribute 0, not NaN"
        );
        assert_eq!(select_best(&g, vec![one.clone()]), one);

        let picked = select_best(&g, vec![one.clone(), sing.clone(), split.clone()]);
        assert_eq!(picked, split, "normalised scalarisation missed the split");
    }
}
