//! The coarsening move: merge whole communities, which no sequence of
//! single-node moves can do.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::graph::CsrGraph;

use super::particle::{Particle, Scratch};

/// Merge adjacent communities whose merge raises CPM at the particle's
/// resolution, best gain first. Returns whether anything moved.
///
/// This operator exists because the node move cannot reach coarse partitions at
/// all. Pulling one vertex out of a 5-clique to join a neighbouring clique
/// loses four internal edges and gains one, so it is rejected at every
/// resolution — yet merging the two cliques outright gains one edge for a
/// penalty of `gamma * 25`, which pays whenever `gamma` is small. Without this,
/// the coarse rungs of the ladder are unreachable and the archive holds only
/// the fine half of the resolution profile.
///
/// Merging `a` and `b` changes CPM by `e_ab - gamma * s_a * s_b`, since the
/// merged community's pair count grows by exactly `s_a * s_b`.
///
/// One sweep accepts a *matching*: no community takes part in two merges. That
/// is what keeps every accepted gain exact — the pair counts and edge counts of
/// disjoint merges do not interact, so the sweep's total gain is the sum of the
/// gains it accepted, with nothing to re-derive. A run of sweeps still reaches
/// full agglomeration, halving the community count at worst per sweep.
pub fn merge_sweep(g: &CsrGraph, p: &mut Particle, s: &mut Scratch) -> bool {
    let mut between: FxHashMap<(i32, i32), u32> = FxHashMap::default();
    for &(u, v) in &g.edges {
        let (a, b) = (p.pos[u as usize], p.pos[v as usize]);
        if a != b {
            *between.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    if between.is_empty() {
        return false;
    }

    let mut cand: Vec<((i32, i32), f64)> = between
        .into_iter()
        .filter_map(|((a, b), e)| {
            let size = |c: i32| f64::from(s.size[c as usize]);
            let gain = f64::from(e) - p.gamma * size(a) * size(b);
            (gain > 0.0).then_some(((a, b), gain))
        })
        .collect();
    if cand.is_empty() {
        return false;
    }
    // Descending gain, ties by the pair, so the order never follows the hash.
    cand.sort_unstable_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });

    let mut taken: FxHashSet<i32> = FxHashSet::default();
    let mut relabel: FxHashMap<i32, i32> = FxHashMap::default();
    for ((a, b), _) in cand {
        if taken.contains(&a) || taken.contains(&b) {
            continue;
        }
        taken.insert(a);
        taken.insert(b);
        let (keep, gone) = if a < b { (a, b) } else { (b, a) };
        relabel.insert(gone, keep);
    }
    if relabel.is_empty() {
        return false;
    }

    for c in p.pos.iter_mut() {
        if let Some(&k) = relabel.get(c) {
            *c = k;
        }
    }
    let (internal, pair_sum) = s.measure(g, &p.pos);
    p.internal = internal;
    p.pair_sum = pair_sum;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::swarm::particle::seeded;
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_cliques};

    fn distinct(p: &Particle) -> usize {
        let mut c = p.pos.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    }

    #[test]
    fn a_low_resolution_collapses_the_ring_that_node_moves_cannot() {
        let g = ring_of_cliques(10, 5);
        let planted: Vec<i32> = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let (mut p, mut s) = seeded(&g, planted, 1e-6);
        let mut rounds = 0;
        while merge_sweep(&g, &mut p, &mut s) && rounds < 20 {
            rounds += 1;
        }
        assert_eq!(distinct(&p), 1, "the ring never collapsed");
        assert_eq!(p.internal, g.m as i64);
    }

    #[test]
    fn a_high_resolution_refuses_every_merge() {
        let g = ring_of_cliques(10, 5);
        let planted: Vec<i32> = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let (mut p, mut s) = seeded(&g, planted.clone(), 1.0);
        assert!(!merge_sweep(&g, &mut p, &mut s));
        assert_eq!(p.pos, planted);
    }

    #[test]
    fn merging_keeps_the_counts_exact() {
        let g = ring_of_cliques(8, 5);
        let planted: Vec<i32> = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        for gamma in [1e-6, 0.005, 0.02, 0.05, 0.2] {
            let (mut p, mut s) = seeded(&g, planted.clone(), gamma);
            for _ in 0..6 {
                if !merge_sweep(&g, &mut p, &mut s) {
                    break;
                }
            }
            let mut check = Scratch::new(g.n);
            assert_eq!(
                check.measure(&g, &p.pos),
                (p.internal, p.pair_sum),
                "gamma {gamma}: the counts left the partition"
            );
        }
    }

    #[test]
    fn a_merge_only_ever_raises_cpm() {
        let g = ring_of_cliques(12, 4);
        let planted: Vec<i32> = (0..g.n).map(|i| (i as i32 / 4) * 4).collect();
        for gamma in [1e-4, 0.01, 0.03, 0.06, 0.1] {
            let (mut p, mut s) = seeded(&g, planted.clone(), gamma);
            let mut before = p.score();
            while merge_sweep(&g, &mut p, &mut s) {
                let after = p.score();
                assert!(
                    after > before - 1e-9,
                    "gamma {gamma}: a merge lowered CPM {before} -> {after}"
                );
                before = after;
            }
        }
    }

    #[test]
    fn the_sweep_is_reproducible() {
        let g = ring_of_cliques(10, 5);
        let planted: Vec<i32> = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let once = || {
            let (mut p, mut s) = seeded(&g, planted.clone(), 0.01);
            merge_sweep(&g, &mut p, &mut s);
            p.pos
        };
        assert_eq!(once(), once());
    }

    #[test]
    fn two_cliques_merge_only_below_the_paying_resolution() {
        // One edge joins them, so the merge gains 1 for a penalty of 25 gamma.
        let g = two_cliques();
        let planted: Vec<i32> = (0..g.n).map(|i| i32::from(i >= 5) * 5).collect();
        for (gamma, expect) in [(0.03, true), (0.05, false)] {
            let (mut p, mut s) = seeded(&g, planted.clone(), gamma);
            assert_eq!(
                merge_sweep(&g, &mut p, &mut s),
                expect,
                "gamma {gamma} decided the wrong way about 1 > 25 gamma"
            );
        }
    }
}
