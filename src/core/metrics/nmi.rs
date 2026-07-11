//! Normalised mutual information (arithmetic mean normalisation).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub fn nmi(mi: f64, hu: f64, hv: f64) -> f64 {
    let mean_h = 0.5 * (hu + hv);
    if mean_h > 0.0 { mi / mean_h } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::super::{contingency, entropy, mutual_info};
    use super::nmi;

    fn score(a: &[i64], b: &[i64]) -> f64 {
        let ct = contingency(a, b);
        nmi(
            mutual_info(&ct),
            entropy(&ct.rows, ct.n),
            entropy(&ct.cols, ct.n),
        )
    }

    #[test]
    fn permuted_labels_are_perfect() {
        assert!((score(&[0, 0, 1, 1, 2, 2, 3, 3], &[1, 1, 0, 0, 3, 3, 2, 2]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn single_cluster_vs_singletons_is_zero() {
        // convention: one side constant, other side splits -> 0
        assert!(score(&[0, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5]).abs() < 1e-12);
    }

    #[test]
    fn both_constant_is_one() {
        assert!((score(&[0, 0, 0], &[5, 5, 5]) - 1.0).abs() < 1e-12);
    }
}
