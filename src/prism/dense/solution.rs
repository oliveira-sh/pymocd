//! Pareto solution plus dominance, q_score, and crowding distance.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::DensePartition;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Solution {
    pub partition: DensePartition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Solution {
    pub fn new(partition: DensePartition) -> Self {
        Solution {
            partition,
            objectives: Vec::with_capacity(3),
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    #[inline(always)]
    pub fn dominates(&self, other: &Solution) -> bool {
        let mut better = false;
        for i in 0..self.objectives.len() {
            let a = self.objectives[i];
            let b = other.objectives[i];
            if a > b {
                return false;
            }
            if a < b {
                better = true;
            }
        }
        better
    }
}

/// Per-rank crowding distance. All archive members share rank = 1.
pub fn crowding_distance(members: &mut [Solution]) {
    let n = members.len();
    if n == 0 {
        return;
    }
    for m in members.iter_mut() {
        m.crowding_distance = 0.0;
    }
    if n <= 2 {
        for m in members.iter_mut() {
            m.crowding_distance = f64::INFINITY;
        }
        return;
    }
    let n_obj = members[0].objectives.len();
    let mut order: Vec<usize> = (0..n).collect();
    for obj_idx in 0..n_obj {
        order.sort_unstable_by(|&a, &b| {
            members[a].objectives[obj_idx]
                .partial_cmp(&members[b].objectives[obj_idx])
                .unwrap_or(Ordering::Equal)
        });
        let first = order[0];
        let last = order[n - 1];
        members[first].crowding_distance = f64::INFINITY;
        members[last].crowding_distance = f64::INFINITY;
        let obj_min = members[first].objectives[obj_idx];
        let obj_max = members[last].objectives[obj_idx];
        let span = obj_max - obj_min;
        if span.abs() <= f64::EPSILON {
            continue;
        }
        let scale = 1.0 / span;
        for k in 1..n - 1 {
            let i = order[k];
            let prev = members[order[k - 1]].objectives[obj_idx];
            let next = members[order[k + 1]].objectives[obj_idx];
            members[i].crowding_distance += (next - prev) * scale;
        }
    }
}

#[inline(always)]
pub fn q_score(s: &Solution) -> f64 {
    (s.objectives.len() as f64) - s.objectives.iter().sum::<f64>()
}
