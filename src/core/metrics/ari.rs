//! Adjusted Rand index (Hubert & Arabie 1985).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Contingency;

pub fn ari(ct: &Contingency) -> f64 {
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let z: f64 = ct.cells.values().map(|&v| c2(v)).sum();
    let rb: f64 = ct.rows.values().map(|&v| c2(v)).sum();
    let cb: f64 = ct.cols.values().map(|&v| c2(v)).sum();
    let m = c2(ct.n);
    let exp = rb * cb / m;
    let max_idx = 0.5 * (rb + cb);
    if (max_idx - exp).abs() < 1e-15 {
        1.0
    } else {
        (z - exp) / (max_idx - exp)
    }
}

#[cfg(test)]
mod tests {
    use super::super::contingency;
    use super::ari;

    fn score(a: &[i64], b: &[i64]) -> f64 {
        ari(&contingency(a, b))
    }

    #[test]
    fn hand_computed_value() {
        // z=2, rb=6, cb=3, exp=6*3/15=1.2, max=4.5 -> (2-1.2)/(4.5-1.2) = 8/33
        assert!((score(&[0, 0, 0, 1, 1, 1], &[0, 0, 1, 1, 2, 2]) - 8.0 / 33.0).abs() < 1e-12);
    }

    #[test]
    fn worse_than_chance_is_negative() {
        assert!(score(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0], &[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) < 0.0);
        assert!(score(&[0, 1, 0, 1, 0, 1, 0, 1], &[0, 0, 1, 1, 2, 2, 3, 3]) < 0.0);
    }

    #[test]
    fn single_cluster_vs_singletons_is_zero() {
        assert!(score(&[0, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5]).abs() < 1e-12);
    }

    #[test]
    fn permuted_labels_are_perfect() {
        assert!((score(&[0, 0, 1, 1, 2, 2, 3, 3], &[1, 1, 0, 0, 3, 3, 2, 2]) - 1.0).abs() < 1e-12);
    }
}
