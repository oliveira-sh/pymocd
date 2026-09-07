//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};

use crate::core::algorithms::rimpso::Labels;
use crate::core::algorithms::rimpso::objectives::Obj;

use super::crowding::crowding;
use super::dominance::dominates;

pub struct Archive {
    obj: Vec<Obj>,
    pos: Vec<Labels>,
    crowd: Vec<f64>,
    cap: usize,
    rungs: Vec<f64>,
}

impl Archive {
    pub fn with_rungs(cap: usize, weights: Vec<f64>) -> Self {
        debug_assert!(!weights.is_empty(), "an archive with no rung to keep");
        let cap = cap.max(1);
        Self {
            obj: Vec::with_capacity(cap + 1),
            pos: Vec::with_capacity(cap + 1),
            crowd: Vec::with_capacity(cap + 1),
            cap,
            rungs: weights,
        }
    }

    #[cfg(test)]
    pub const fn len(&self) -> usize {
        self.obj.len()
    }

    #[cfg(test)]
    pub fn objectives(&self) -> &[Obj] {
        &self.obj
    }

    pub fn into_parts(self) -> (Vec<Labels>, Vec<Obj>) {
        (self.pos, self.obj)
    }

    pub fn offer(&mut self, obj: Obj, pos: &[i32]) -> bool {
        for existing in &self.obj {
            if dominates(existing, &obj) || *existing == obj {
                return false;
            }
        }
        let mut w = 0;
        for r in 0..self.obj.len() {
            if !dominates(&obj, &self.obj[r]) {
                if w != r {
                    self.obj.swap(w, r);
                    self.pos.swap(w, r);
                }
                w += 1;
            }
        }
        self.obj.truncate(w);
        self.pos.truncate(w);
        self.obj.push(obj);
        self.pos.push(pos.to_vec());
        true
    }

    pub fn prune(&mut self) {
        if self.obj.len() > self.cap {
            let n = self.obj.len();
            let mut keep = vec![false; n];
            for &w in &self.rungs {
                let mut best = 0usize;
                let mut bv = f64::INFINITY;
                for i in 0..n {
                    let v = self.obj[i][0] + w * self.obj[i][1];
                    if v < bv {
                        bv = v;
                        best = i;
                    }
                }
                keep[best] = true;
            }
            let mut it = keep.iter();
            self.obj.retain(|_| *it.next().unwrap());
            let mut it = keep.iter();
            self.pos.retain(|_| *it.next().unwrap());
        }
        self.crowd = crowding(&self.obj);
    }

    pub fn leader(&self, r: &mut impl Rng) -> usize {
        let n = self.obj.len();
        debug_assert!(n > 0, "leader() on an empty archive");
        if n == 1 {
            return 0;
        }
        let i = r.random_range(0..n);
        let j = r.random_range(0..n);
        if self.crowd[i] > self.crowd[j] || (self.crowd[i] == self.crowd[j] && i <= j) {
            i
        } else {
            j
        }
    }

    pub fn position(&self, i: usize) -> &Labels {
        &self.pos[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::rimpso::config::defaults::DEFAULT_SEED;
    use crate::core::algorithms::rimpso::utils::sampling::slot_rng;

    fn rungs(k: usize) -> Vec<f64> {
        (0..k)
            .map(|i| 10f64.powf(i as f64 / (k.max(2) - 1) as f64 * 4.0 - 2.0))
            .collect()
    }

    fn seeded(cap: usize, pts: &[Obj]) -> Archive {
        let mut a = Archive::with_rungs(cap, rungs(cap));
        for (i, &o) in pts.iter().enumerate() {
            a.offer(o, &[i as i32]);
        }
        a
    }

    #[test]
    fn dominated_candidates_are_refused() {
        let mut a = seeded(8, &[[0.2, 0.2]]);
        assert!(!a.offer([0.3, 0.3], &[9]), "a dominated point was accepted");
        assert!(!a.offer([0.2, 0.2], &[9]), "a duplicate point was accepted");
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn a_dominating_candidate_evicts_the_members_it_beats() {
        let mut a = seeded(8, &[[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]);
        assert_eq!(a.len(), 3);
        assert!(a.offer([0.4, 0.4], &[9]));
        assert_eq!(a.len(), 3, "the dominated member survived");
        assert!(!a.objectives().contains(&[0.5, 0.5]));
        assert!(a.objectives().contains(&[0.4, 0.4]));
    }

    #[test]
    fn every_member_stays_mutually_nondominated() {
        let mut a = Archive::with_rungs(64, rungs(64));
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..400 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((s >> 33) % 1000) as f64 / 1000.0;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = ((s >> 33) % 1000) as f64 / 1000.0;
            a.offer([x, y], &[i]);
        }
        a.prune();
        let o = a.objectives();
        for i in 0..o.len() {
            for j in 0..o.len() {
                assert!(
                    i == j || !dominates(&o[i], &o[j]),
                    "{:?} dominates {:?}",
                    o[i],
                    o[j]
                );
            }
        }
        assert!(o.len() <= 64);
    }

    #[test]
    fn the_rung_prune_keeps_the_best_member_at_each_resolution() {
        let weights = vec![0.05, 1.0, 20.0];
        let mut a = Archive::with_rungs(3, weights.clone());
        for i in 0..40i32 {
            let t = f64::from(i) / 39.0;
            a.offer([t, 1.0 - t], &[i]);
        }
        a.prune();
        assert!(a.len() <= 3, "the rung prune exceeded its cap: {}", a.len());
        let o = a.objectives().to_vec();
        for &w in &weights {
            let kept = o
                .iter()
                .map(|p| p[0] + w * p[1])
                .fold(f64::INFINITY, f64::min);
            let ideal = (0..40)
                .map(|i| {
                    let t = f64::from(i) / 39.0;
                    t + w * (1.0 - t)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                kept <= ideal + 1e-12,
                "rung w={w} lost its optimum: kept {kept}, best {ideal}"
            );
        }
    }

    #[test]
    fn leader_selection_is_reproducible() {
        let mut a = Archive::with_rungs(8, rungs(8));
        for i in 0..8i32 {
            let t = f64::from(i) / 7.0;
            a.offer([t, 1.0 - t], &[i]);
        }
        a.prune();
        let draw = || {
            let mut r = slot_rng(DEFAULT_SEED, 3, 5);
            (0..32).map(|_| a.leader(&mut r)).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }
}
