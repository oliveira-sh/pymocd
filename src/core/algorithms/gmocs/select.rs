//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>

use rustc_hash::FxHashMap;

use crate::core::algorithms::gmocs::Labels;
use crate::core::graph::CsrGraph;

pub(crate) fn select_best(
    g: &CsrGraph,
    front: Vec<Labels>,
    dev: &mut crate::core::algorithms::gmocs::gpu::Gpu,
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
    pick_best(front, &obj)
}

fn pick_best(front: Vec<Labels>, obj: &[[f64; 4]]) -> Labels {
    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
    for o in obj {
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

    fn objs(g: &CsrGraph, front: &[Labels]) -> Vec<[f64; 4]> {
        front
            .iter()
            .map(|p| {
                let (kkm, rc) = crate::core::algorithms::gmocs::objectives::kkm_rc(g, p);
                let (intra, inter) =
                    crate::core::algorithms::gmocs::objectives::intra_inter(g, p);
                [kkm, rc, intra, inter]
            })
            .collect()
    }

    #[test]
    fn selector_normalises_and_breaks_ties_by_lowest_index() {
        let nodes: Vec<i32> = (0..10).collect();
        let g = CsrGraph::from_edges(&nodes, &two_clique_edges());

        let split: Labels = (0..g.n).map(|i| i32::from(i >= 5)).collect();
        let one: Labels = vec![0; g.n];
        let sing: Labels = (0..g.n as i32).collect();

        let f = vec![split.clone(), split.clone()];
        let o = objs(&g, &f);
        assert_eq!(
            pick_best(f, &o),
            split,
            "a constant column must contribute 0, not NaN"
        );
        let f = vec![one.clone()];
        let o = objs(&g, &f);
        assert_eq!(pick_best(f, &o), one);

        let f = vec![one, sing, split.clone()];
        let o = objs(&g, &f);
        assert_eq!(pick_best(f, &o), split, "normalised scalarisation missed the split");
    }
}
