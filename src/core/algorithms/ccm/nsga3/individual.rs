//! A population member and the non-dominated sort that ranks the population.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::ccm::locus::{self, Genome};

/// `objectives` are minimized (see `objectives::evaluate` for the sign
/// convention CCM feeds in); `rank` is 1-based, 1 = non-dominated.
#[derive(Clone)]
pub struct Individual {
    pub genome: Genome,
    pub labels: Vec<i32>,
    pub objectives: Vec<f64>,
    pub rank: usize,
}

impl Individual {
    #[inline]
    fn dominates(&self, other: &Self) -> bool {
        let mut strictly_better = false;
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                strictly_better = true;
            }
        }
        strictly_better
    }
}

pub fn new_individual(genome: Genome) -> Individual {
    let labels = locus::decode(&genome);
    Individual {
        genome,
        labels,
        objectives: Vec::new(),
        rank: usize::MAX,
    }
}

/// Deb et al. 2002, Alg. 1.
pub fn fast_non_dominated_sort(pop: &mut [Individual]) {
    let n = pop.len();
    if n == 0 {
        return;
    }

    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_count: Vec<usize> = vec![0; n];

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
    }

    let mut front: Vec<usize> = (0..n).filter(|&i| dom_count[i] == 0).collect();
    let mut rank = 1usize;
    while !front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &front {
            pop[i].rank = rank;
            for &j in &dominated[i] {
                dom_count[j] -= 1;
                if dom_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        rank += 1;
        front = next_front;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominates_minimization_convention() {
        let mut a = new_individual(vec![0]);
        let mut b = a.clone();
        a.objectives = vec![-1.0, -1.0];
        b.objectives = vec![-1.0, -0.5];
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn fast_non_dominated_sort_ranks() {
        let mut pop: Vec<Individual> = vec![
            new_individual(vec![0]),
            new_individual(vec![0]),
            new_individual(vec![0]),
        ];
        pop[0].objectives = vec![0.0, 0.0];
        pop[1].objectives = vec![1.0, 1.0];
        pop[2].objectives = vec![2.0, 2.0];
        fast_non_dominated_sort(&mut pop);
        assert_eq!(pop[0].rank, 1);
        assert_eq!(pop[1].rank, 2);
        assert_eq!(pop[2].rank, 3);
    }
}
