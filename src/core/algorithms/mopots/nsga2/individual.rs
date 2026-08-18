//! The mopots population member: a partition, its objective vector and its NSGA-II rank.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Partition;

pub const TOURNAMENT_SIZE: usize = 2;

/// Objective vector, always `[cut, pair]`, both minimised.
pub type ObjVec = Vec<f64>;

#[derive(Clone, Debug)]
pub struct Individual {
    pub partition: Partition,
    pub objectives: ObjVec,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Individual {
    pub fn new(partition: Partition) -> Self {
        Self {
            partition,
            objectives: vec![0.0, 0.0],
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    /// True when self is no worse on every objective and strictly better on one.
    #[inline(always)]
    pub fn dominates(&self, other: &Self) -> bool {
        let mut at_least_one_better = false;

        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }
}
