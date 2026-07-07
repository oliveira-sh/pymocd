//! Individual representation + Pareto dominance + fast non-dominated sort for
//! the self-contained NSGA-III-KRM engine. The genome is locus-based (see
//! `locus.rs`); objectives are stored all-minimized as `[KKM, RC, -Q]` so a
//! single `dominates` rule works across the bi-min/uni-max objective mix.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::Genome;
use crate::core::graph::Partition;

#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: Genome,
    pub partition: Partition,
    /// `[KKM, RC, -Q]`, all minimized.
    pub objectives: Vec<f64>,
    pub rank: usize,
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

/// Sequential fast non-dominated sort (Deb et al. 2002). Deliberately a plain
/// `O(n^2)` scan with no data-parallel iterators or atomics -- single-threaded
/// by construction, so this engine's cost reflects the paper's method rather
/// than this repo's optimized (parallel) sort.
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
