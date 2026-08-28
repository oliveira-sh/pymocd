//! Adjusted Rand index (Hubert & Arabie 1985).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Contingency;

/// Adjusted Rand index: the pair-counting agreement, corrected for chance.
/// Returns 1.0 in the degenerate case where the maximum index equals the
/// expected one (both labelings all-singletons or all-one-cluster).
pub fn ari(ct: &Contingency) -> f64 {
    let pairs = |x: f64| x * (x - 1.0) / 2.0;
    let index: f64 = ct.cells.values().map(|&v| pairs(v)).sum();
    let row_pairs: f64 = ct.rows.values().map(|&v| pairs(v)).sum();
    let col_pairs: f64 = ct.cols.values().map(|&v| pairs(v)).sum();
    let expected = row_pairs * col_pairs / pairs(ct.n);
    let max_index = 0.5 * (row_pairs + col_pairs);
    if (max_index - expected).abs() < 1e-15 {
        1.0
    } else {
        (index - expected) / (max_index - expected)
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
        // index=2, row_pairs=6, col_pairs=3, expected=6*3/15=1.2, max=4.5
        // -> (2 - 1.2) / (4.5 - 1.2) = 8/33
        assert!((score(&[0, 0, 0, 1, 1, 1], &[0, 0, 1, 1, 2, 2]) - 8.0 / 33.0).abs() < 1e-12);
    }

    #[test]
    fn worse_than_chance_is_negative() {
        assert!(
            score(
                &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                &[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
            ) < 0.0
        );
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
