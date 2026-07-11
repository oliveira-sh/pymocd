//! Contingency table between two label vectors + marginal entropy.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

pub struct Contingency {
    pub cells: FxHashMap<(i64, i64), f64>,
    pub rows: FxHashMap<i64, f64>,
    pub cols: FxHashMap<i64, f64>,
    pub n: f64,
}

pub fn contingency(a: &[i64], b: &[i64]) -> Contingency {
    let mut cells: FxHashMap<(i64, i64), f64> = FxHashMap::default();
    let mut rows: FxHashMap<i64, f64> = FxHashMap::default();
    let mut cols: FxHashMap<i64, f64> = FxHashMap::default();
    for (&x, &y) in a.iter().zip(b) {
        *cells.entry((x, y)).or_insert(0.0) += 1.0;
        *rows.entry(x).or_insert(0.0) += 1.0;
        *cols.entry(y).or_insert(0.0) += 1.0;
    }
    Contingency {
        cells,
        rows,
        cols,
        n: a.len() as f64,
    }
}

pub fn entropy(marg: &FxHashMap<i64, f64>, n: f64) -> f64 {
    marg.values()
        .filter(|&&c| c > 0.0)
        .map(|&c| {
            let p = c / n;
            -p * p.ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_marginals() {
        // table: [[2,1,0],[0,1,2]]
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];
        let ct = contingency(&a, &b);
        assert_eq!(ct.n, 6.0);
        assert_eq!(ct.cells[&(0, 0)], 2.0);
        assert_eq!(ct.cells[&(0, 1)], 1.0);
        assert_eq!(ct.cells[&(1, 1)], 1.0);
        assert_eq!(ct.cells[&(1, 2)], 2.0);
        assert_eq!(ct.cells.len(), 4); // zero cells absent
        assert_eq!(ct.rows[&0], 3.0);
        assert_eq!(ct.rows[&1], 3.0);
        assert_eq!(ct.cols[&0], 2.0);
        assert_eq!(ct.cols[&1], 2.0);
        assert_eq!(ct.cols[&2], 2.0);
    }

    #[test]
    fn entropy_of_k_equal_clusters_is_ln_k() {
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];
        let ct = contingency(&a, &b);
        assert!((entropy(&ct.rows, ct.n) - 2.0_f64.ln()).abs() < 1e-12);
        assert!((entropy(&ct.cols, ct.n) - 3.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn entropy_single_cluster_is_zero() {
        let a = vec![0, 0, 0, 0, 0, 0];
        let b = vec![0, 1, 2, 3, 4, 5];
        let ct = contingency(&a, &b);
        assert_eq!(entropy(&ct.rows, ct.n), 0.0);
        assert!((entropy(&ct.cols, ct.n) - 6.0_f64.ln()).abs() < 1e-12);
    }
}
