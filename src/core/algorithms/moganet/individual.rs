//! Individual + Pareto dominance + fast non-dominated sort for MOGA-Net.
//! Objectives are stored as `[-CS, -CF]` (both minimized) since Pizzuti's CS
//! and CF are both maximized.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::Genome;
use crate::core::graph::Partition;

#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: Genome,
    pub partition: Partition,
    /// `[-CS, -CF]`, both minimized.
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Individual {
    #[inline]
    pub fn dominates(&self, other: &Individual) -> bool {
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

/// Fast non-dominated sort (Deb et al. 2002, NSGA-II); deliberately
/// sequential.
pub fn fast_non_dominated_sort(pop: &mut [Individual]) {
    let n = pop.len();
    if n == 0 {
        return;
    }

    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_count = vec![0usize; n];
    let mut front: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if pop[i].dominates(&pop[j]) {
                dominated[i].push(j);
            } else if pop[j].dominates(&pop[i]) {
                dom_count[i] += 1;
            }
        }
        if dom_count[i] == 0 {
            pop[i].rank = 1;
            front.push(i);
        }
    }

    let mut rank = 1usize;
    while !front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &front {
            for &j in &dominated[i] {
                dom_count[j] -= 1;
                if dom_count[j] == 0 {
                    pop[j].rank = rank + 1;
                    next_front.push(j);
                }
            }
        }
        rank += 1;
        front = next_front;
    }
}

/// NSGA-II crowding distance (Deb et al. 2002) within each rank front: per
/// objective, sort the front and accumulate normalized neighbour gaps;
/// boundary individuals get `+inf`. Assumes `fast_non_dominated_sort` has
/// already assigned `rank`.
pub fn calculate_crowding_distance(pop: &mut [Individual]) {
    if pop.is_empty() {
        return;
    }
    let n_obj = pop[0].objectives.len();
    for ind in pop.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    let max_rank = pop.iter().map(|i| i.rank).max().unwrap_or(0);
    for rank in 1..=max_rank {
        let mut indices: Vec<usize> = (0..pop.len()).filter(|&i| pop[i].rank == rank).collect();
        if indices.is_empty() {
            continue;
        }
        if indices.len() <= 2 {
            for &i in &indices {
                pop[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }
        for obj in 0..n_obj {
            indices.sort_unstable_by(|&a, &b| {
                pop[a].objectives[obj]
                    .partial_cmp(&pop[b].objectives[obj])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            pop[indices[0]].crowding_distance = f64::INFINITY;
            pop[indices[indices.len() - 1]].crowding_distance = f64::INFINITY;

            let obj_min = pop[indices[0]].objectives[obj];
            let obj_max = pop[indices[indices.len() - 1]].objectives[obj];
            if (obj_max - obj_min).abs() > f64::EPSILON {
                let scale = 1.0 / (obj_max - obj_min);
                for w in 1..indices.len() - 1 {
                    let prev = pop[indices[w - 1]].objectives[obj];
                    let next = pop[indices[w + 1]].objectives[obj];
                    pop[indices[w]].crowding_distance += (next - prev) * scale;
                }
            }
        }
    }
}
