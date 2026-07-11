//! Pairwise F1: harmonic mean of pair-precision and pair-recall over
//! same-community node pairs.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Contingency;

pub fn f1(ct: &Contingency) -> f64 {
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let tp: f64 = ct.cells.values().map(|&v| c2(v)).sum();
    let pairs_a: f64 = ct.rows.values().map(|&v| c2(v)).sum();
    let pairs_b: f64 = ct.cols.values().map(|&v| c2(v)).sum();
    if pairs_a + pairs_b == 0.0 {
        return 1.0; // both all-singletons: trivially identical
    }
    2.0 * tp / (pairs_a + pairs_b)
}

#[cfg(test)]
mod tests {
    use super::super::contingency;
    use super::*;

    fn score(a: &[i64], b: &[i64]) -> f64 {
        f1(&contingency(a, b))
    }

    #[test]
    fn hand_computed_value() {
        // tp=2, pairs_a=6, pairs_b=3 -> 2*2/(6+3) = 4/9
        assert!((score(&[0, 0, 0, 1, 1, 1], &[0, 0, 1, 1, 2, 2]) - 4.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn identical_and_permuted_are_perfect() {
        let a = vec![0, 0, 1, 1];
        assert!((score(&a, &a) - 1.0).abs() < 1e-12);
        assert!((score(&[0, 0, 1, 1, 2, 2, 3, 3], &[1, 1, 0, 0, 3, 3, 2, 2]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_shared_pairs_is_zero() {
        assert!(score(&[0, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5]).abs() < 1e-12);
        assert!(score(&[0, 1, 0, 1, 0, 1, 0, 1], &[0, 0, 1, 1, 2, 2, 3, 3]).abs() < 1e-12);
    }
}
