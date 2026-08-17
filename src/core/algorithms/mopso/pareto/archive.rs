//! The external Pareto archive: the swarm's shared memory of the best
//! trade-offs it has seen, and the pool leaders are drawn from.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::objectives::Obj;

use super::crowding::crowding;
use super::dominance::dominates;

/// A bounded archive of mutually non-dominated partitions.
///
/// Every member is a point on the graph's resolution profile: because CPM at
/// any `gamma` is a weighted sum of the two objectives, the archive is the
/// swarm's running approximation of the whole `gamma` sweep. Truncation is by
/// crowding distance, which spreads the survivors along that profile instead of
/// letting the swarm pile up at one granularity.
pub struct Archive {
    obj: Vec<Obj>,
    pos: Vec<Labels>,
    crowd: Vec<f64>,
    cap: usize,
}

impl Archive {
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            obj: Vec::with_capacity(cap + 1),
            pos: Vec::with_capacity(cap + 1),
            crowd: Vec::with_capacity(cap + 1),
            cap,
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    #[cfg(test)]
    pub fn objectives(&self) -> &[Obj] {
        &self.obj
    }

    pub fn into_parts(self) -> (Vec<Labels>, Vec<Obj>) {
        (self.pos, self.obj)
    }

    /// Offer a candidate. Rejected when an incumbent dominates it or already
    /// occupies its exact objective point; otherwise it evicts everything it
    /// dominates and joins. `pos` is only cloned on acceptance, which is what
    /// keeps a swarm-wide offer pass off the allocator at large `n`.
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

    /// Drop the most crowded members down to the capacity, then refresh the
    /// crowding distances leader selection reads. Called once per iteration,
    /// after the whole swarm has been offered.
    pub fn prune(&mut self) {
        if self.obj.len() > self.cap {
            let d = crowding(&self.obj);
            let mut order: Vec<u32> = (0..self.obj.len() as u32).collect();
            // Least crowded first; index breaks ties so the survivors never
            // depend on the sort's stability.
            order.sort_unstable_by(|&a, &b| {
                d[b as usize]
                    .partial_cmp(&d[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut keep = vec![false; self.obj.len()];
            for &i in &order[..self.cap] {
                keep[i as usize] = true;
            }
            let mut it = keep.iter();
            self.obj.retain(|_| *it.next().unwrap());
            let mut it = keep.iter();
            self.pos.retain(|_| *it.next().unwrap());
        }
        self.crowd = crowding(&self.obj);
    }

    /// Binary tournament on crowding distance: the leader is drawn from the
    /// sparse end of the front, which is where the swarm still has profile left
    /// to explore.
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
    use crate::core::algorithms::mopso::utils::sampling::slot_rng;

    fn seeded(cap: usize, pts: &[Obj]) -> Archive {
        let mut a = Archive::with_capacity(cap);
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
        let mut a = Archive::with_capacity(64);
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
    fn pruning_keeps_both_extremes() {
        let mut a = Archive::with_capacity(3);
        for i in 0..40i32 {
            let t = f64::from(i) / 39.0;
            a.offer([t, 1.0 - t], &[i]);
        }
        a.prune();
        assert_eq!(a.len(), 3);
        let o = a.objectives();
        assert!(o.contains(&[0.0, 1.0]), "the fine extreme was truncated");
        assert!(o.contains(&[1.0, 0.0]), "the coarse extreme was truncated");
    }

    #[test]
    fn leader_selection_is_reproducible() {
        let mut a = Archive::with_capacity(8);
        for i in 0..8i32 {
            let t = f64::from(i) / 7.0;
            a.offer([t, 1.0 - t], &[i]);
        }
        a.prune();
        let draw = || {
            let mut r = slot_rng(3, 5);
            (0..32).map(|_| a.leader(&mut r)).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }
}
