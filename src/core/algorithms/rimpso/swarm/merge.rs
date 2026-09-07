//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::particle::{Particle, Scratch};

#[inline]
fn rank(gain: f64, a: u32, b: u32) -> u128 {
    (u128::from(!gain.to_bits()) << 64) | (u128::from(a) << 32) | u128::from(b)
}

pub fn merge_sweep(g: &CsrGraph, p: &mut Particle, s: &mut Scratch) -> bool {
    s.bucket_by_community(&p.pos);

    s.cand.clear();
    for a in 0..g.n {
        let sa = s.size[a];
        if sa == 0 {
            continue;
        }
        let lo = s.start[a] as usize;
        for k in lo..lo + sa as usize {
            for &v in g.neighbors(s.bucket[k] as usize) {
                let b = p.pos[v as usize] as usize;
                if b > a {
                    if s.link[b] == 0 {
                        s.touched.push(b as u32);
                    }
                    s.link[b] += 1;
                }
            }
        }
        let penalty = p.gamma * f64::from(sa);
        for &bi in &s.touched {
            let b = bi as usize;
            let e = s.link[b];
            s.link[b] = 0;
            let gain = f64::from(e) - penalty * f64::from(s.size[b]);
            if gain > 0.0 {
                s.cand.push(rank(gain, a as u32, b as u32));
            }
        }
        s.touched.clear();
    }
    if s.cand.is_empty() {
        return false;
    }
    s.cand.sort_unstable();

    for k in &s.cand {
        let (a, b) = ((*k >> 32) as u32 as usize, *k as u32 as usize);
        if s.link[a] != 0 || s.link[b] != 0 {
            continue;
        }
        s.link[a] = 1;
        s.link[b] = 1;
        s.touched.push(a as u32);
        s.touched.push(b as u32);
        s.bucket[a] = a as u32;
        s.bucket[b] = a as u32;
    }

    for c in &mut p.pos {
        let ci = *c as usize;
        if s.link[ci] != 0 {
            *c = s.bucket[ci] as i32;
        }
    }
    for &c in &s.touched {
        s.link[c as usize] = 0;
    }
    s.touched.clear();

    let (internal, pair_sum) = s.measure(g, &p.pos);
    p.internal = internal;
    p.pair_sum = pair_sum;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::rimpso::swarm::particle::seeded;
    use crate::core::algorithms::rimpso::utils::fixtures::{ring_of_cliques, two_cliques};

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
    fn two_cliques_merge_only_below_the_paying_resolution() {
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
