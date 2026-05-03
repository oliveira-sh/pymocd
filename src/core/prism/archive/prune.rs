//! Crowding-distance refresh and capacity pruning.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Archive;
use crate::core::prism::dense::crowding_distance;
use std::cmp::Ordering;

impl Archive {
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
}
