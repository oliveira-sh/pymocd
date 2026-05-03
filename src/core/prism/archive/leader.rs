//! Binary-tournament leader selection by crowding distance.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Archive;
use crate::core::prism::dense::Solution;
use rand::RngExt;

impl Archive {
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
}
