//! Fast non-dominated sorting (Deb et al. 2002), with the domination matrix built in parallel.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::individual::Individual;

pub fn fast_non_dominated_sort(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }

    let n = population.len();
    let mut fronts: Vec<Vec<usize>> = Vec::with_capacity(n / 2);
    fronts.push(Vec::with_capacity(n / 2));

    let mut dominated_data = Vec::new();
    let mut dominated_ranges = Vec::with_capacity(n);
    let domination_count: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();

    let domination_relations: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::new();
            let mut count = 0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                if population[i].dominates(&population[j]) {
                    dominated.push(j);
                } else if population[j].dominates(&population[i]) {
                    count += 1;
                }
            }

            (dominated, count)
        })
        .collect();

    for (i, (dominated, count)) in domination_relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        dominated_ranges.push(start..dominated_data.len());
        domination_count[i].store(count, Ordering::Relaxed);

        if count == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }

    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = &fronts[front_idx];
        let next_front: Vec<usize> = current_front
            .par_iter()
            .fold(Vec::new, |mut acc, &i| {
                let range = &dominated_ranges[i];
                for &j in &dominated_data[range.start..range.end] {
                    let prev = domination_count[j].fetch_sub(1, Ordering::Relaxed);
                    if prev == 1 {
                        acc.push(j);
                    }
                }
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        front_idx += 1;
        if !next_front.is_empty() {
            for &j in &next_front {
                population[j].rank = front_idx + 1;
            }
            fronts.push(next_front);
        } else {
            break;
        }
    }
}
