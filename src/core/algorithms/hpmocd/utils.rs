//! Max-modularity selector for HP-MOCD: assumes the intra/inter objective
//! encoding, `Q = n − Σ objectives`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::Individual;
use std::cmp::Ordering;

#[inline(always)]
pub fn q(ind: &Individual) -> f64 {
    let n: f64 = ind.objectives.len() as f64;
    n - ind.objectives.iter().sum::<f64>()
}
#[inline(always)]
pub fn max_q_selection(population: &[Individual]) -> &Individual {
    population
        .iter()
        .max_by(|a, b| q(a).partial_cmp(&q(b)).unwrap_or(Ordering::Equal))
        .expect("Empty population")
}
