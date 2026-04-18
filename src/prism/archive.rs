//! smpso/archive.rs
//! External Archive over SMPSO-local `Solution` (no HP-MOCD `Individual`, no hashmap partition).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::ufop::dense::{Solution, crowding_distance};
use rand::RngExt;
use std::cmp::Ordering;

pub struct Archive {
    pub members: Vec<Solution>,
    pub capacity: usize,
}

impl Archive {
    pub fn new(capacity: usize) -> Self {
        Archive {
            members: Vec::with_capacity(capacity + capacity / 4 + 1),
            capacity,
        }
    }

    pub fn try_add(&mut self, candidate: Solution) -> bool {
        let mut to_drop: Vec<usize> = Vec::new();
        for (i, m) in self.members.iter().enumerate() {
            if m.dominates(&candidate) {
                return false;
            }
            if candidate.dominates(m) {
                to_drop.push(i);
            } else if m.objectives == candidate.objectives {
                return false;
            }
        }
        for &i in to_drop.iter().rev() {
            self.members.swap_remove(i);
        }
        self.members.push(candidate);
        true
    }

    pub fn refresh_crowding(&mut self) {
        if self.members.is_empty() {
            return;
        }
        for m in self.members.iter_mut() {
            m.rank = 1;
        }
        crowding_distance(&mut self.members);
    }

    pub fn prune(&mut self) {
        self.refresh_crowding();
        if self.members.len() <= self.capacity {
            return;
        }
        self.members.sort_unstable_by(|a, b| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        });
        self.members.truncate(self.capacity);
    }

    pub fn prune_if_over(&mut self, buffer: usize) {
        self.refresh_crowding();
        if self.members.len() <= self.capacity + buffer {
            return;
        }
        self.members.sort_unstable_by(|a, b| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        });
        self.members.truncate(self.capacity);
    }

    pub fn select_leader(&self) -> &Solution {
        let n = self.members.len();
        debug_assert!(n > 0, "select_leader on empty archive");
        let mut rng = rand::rng();
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let a = &self.members[i];
        let b = &self.members[j];
        if a.crowding_distance >= b.crowding_distance {
            a
        } else {
            b
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }
}
