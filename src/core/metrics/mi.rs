//! Mutual information of a contingency table (shared by NMI and AMI).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Contingency;

pub fn mutual_info(ct: &Contingency) -> f64 {
    let n = ct.n;
    ct.cells
        .iter()
        .map(|(&(i, j), &nij)| {
            let ai = ct.rows[&i];
            let bj = ct.cols[&j];
            (nij / n) * ((n * nij) / (ai * bj)).ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::super::contingency::contingency;
    use super::*;

    fn mi(a: &[i64], b: &[i64]) -> f64 {
        mutual_info(&contingency(a, b))
    }

    #[test]
    fn identical_k_cluster_labels_give_ln_k() {
        let a = [0, 0, 1, 1];
        assert!((mi(&a, &a) - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn permuted_labels_give_full_entropy() {
        let a = [0, 0, 1, 1, 2, 2, 3, 3];
        let b = [1, 1, 0, 0, 3, 3, 2, 2];
        assert!((mi(&a, &b) - 4.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn independent_labelings_give_zero() {
        let a = [0, 1, 0, 1, 0, 1, 0, 1];
        let b = [0, 0, 1, 1, 2, 2, 3, 3];
        assert!(mi(&a, &b).abs() < 1e-12);
    }
}
