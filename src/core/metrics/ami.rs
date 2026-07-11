//! Adjusted mutual information (Vinh, Epps & Bailey 2010).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Contingency;
use crate::core::utils::special::gammln;

fn ln_fact(x: f64) -> f64 {
    gammln(x + 1.0)
}

/// Exact expected mutual information under the permutation model,
/// log-space hypergeometric weights.
fn expected_mi(ct: &Contingency) -> f64 {
    let n = ct.n;
    let ln_n_fact = ln_fact(n);
    let mut emi = 0.0;
    for &ai in ct.rows.values() {
        for &bj in ct.cols.values() {
            let lo = (ai + bj - n).max(1.0);
            let hi = ai.min(bj);
            let mut nij = lo;
            while nij <= hi {
                let ln_w = ln_fact(ai) + ln_fact(bj) + ln_fact(n - ai) + ln_fact(n - bj)
                    - ln_n_fact
                    - ln_fact(nij)
                    - ln_fact(ai - nij)
                    - ln_fact(bj - nij)
                    - ln_fact(n - ai - bj + nij);
                emi += (nij / n) * ((n * nij) / (ai * bj)).ln() * ln_w.exp();
                nij += 1.0;
            }
        }
    }
    emi
}

pub fn ami(ct: &Contingency, mi: f64, hu: f64, hv: f64) -> f64 {
    if hu == 0.0 && hv == 0.0 {
        return 1.0;
    }
    let emi = expected_mi(ct);
    let denom = 0.5 * (hu + hv) - emi;
    if denom.abs() < 1e-15 {
        0.0
    } else {
        (mi - emi) / denom
    }
}

#[cfg(test)]
mod tests {
    use super::super::{contingency, entropy, mutual_info};
    use super::ami;

    fn score(a: &[i64], b: &[i64]) -> f64 {
        let ct = contingency(a, b);
        ami(
            &ct,
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
    fn worse_than_chance_is_negative() {
        // chance-adjusted: crossing labelings must score below zero
        assert!(score(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0], &[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) < 0.0);
        assert!(score(&[0, 1, 0, 1, 0, 1, 0, 1], &[0, 0, 1, 1, 2, 2, 3, 3]) < 0.0);
    }

    #[test]
    fn single_cluster_vs_singletons_is_zero() {
        assert!(score(&[0, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5]).abs() < 1e-12);
    }
}
