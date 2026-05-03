//! try_add: linear dominance scan over current members.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Archive;
use crate::core::prism::dense::Solution;

impl Archive {
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
}
