//! Ground-truth agreement metrics between two label vectors: NMI, AMI, ARI.
//! Exact implementations (natural log, arithmetic normalisation — matching
//! scikit-learn's defaults; AMI uses the exact Vinh et al. (2010) expected
//! mutual information with log-space hypergeometric terms).

use rustc_hash::FxHashMap;

use super::special::gammln;

fn ln_fact(x: f64) -> f64 {
    gammln(x + 1.0)
}

struct Contingency {
    cells: FxHashMap<(i64, i64), f64>,
    rows: FxHashMap<i64, f64>,
    cols: FxHashMap<i64, f64>,
    n: f64,
}

fn contingency(a: &[i64], b: &[i64]) -> Contingency {
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

fn entropy(marg: &FxHashMap<i64, f64>, n: f64) -> f64 {
    marg.values()
        .filter(|&&c| c > 0.0)
        .map(|&c| {
            let p = c / n;
            -p * p.ln()
        })
        .sum()
}

fn mutual_info(ct: &Contingency) -> f64 {
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

/// Exact expected mutual information under the permutation model
/// (Vinh, Epps & Bailey 2010), log-space hypergeometric weights.
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

/// (NMI, AMI, ARI) between two label vectors of equal length.
/// NMI/AMI use arithmetic mean normalisation (scikit-learn default).
pub fn gt_metrics(a: &[i64], b: &[i64]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let ct = contingency(a, b);
    let n = ct.n;
    let hu = entropy(&ct.rows, n);
    let hv = entropy(&ct.cols, n);
    let mi = mutual_info(&ct);
    let mean_h = 0.5 * (hu + hv);

    let nmi = if mean_h > 0.0 { mi / mean_h } else { 1.0 };

    let ami = if hu == 0.0 && hv == 0.0 {
        1.0
    } else {
        let emi = expected_mi(&ct);
        let denom = mean_h - emi;
        if denom.abs() < 1e-15 {
            0.0
        } else {
            (mi - emi) / denom
        }
    };

    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let z: f64 = ct.cells.values().map(|&v| c2(v)).sum();
    let rb: f64 = ct.rows.values().map(|&v| c2(v)).sum();
    let cb: f64 = ct.cols.values().map(|&v| c2(v)).sum();
    let m = c2(n);
    let exp = rb * cb / m;
    let max_idx = 0.5 * (rb + cb);
    let ari = if (max_idx - exp).abs() < 1e-15 {
        1.0
    } else {
        (z - exp) / (max_idx - exp)
    };

    (nmi, ami, ari)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_labelings_are_perfect() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let (nmi, ami, ari) = gt_metrics(&a, &a);
        assert!((nmi - 1.0).abs() < 1e-12);
        assert!((ami - 1.0).abs() < 1e-12);
        assert!((ari - 1.0).abs() < 1e-12);
    }

    #[test]
    fn permuted_labels_are_perfect() {
        let a = vec![0, 0, 1, 1];
        let b = vec![7, 7, 3, 3];
        let (nmi, ami, ari) = gt_metrics(&a, &b);
        assert!((nmi - 1.0).abs() < 1e-12);
        assert!((ami - 1.0).abs() < 1e-12);
        assert!((ari - 1.0).abs() < 1e-12);
    }

    #[test]
    fn known_sklearn_values() {
        // sklearn: nmi=0.5158, ami=0.2987, ari=0.2424 for this pair
        let a = vec![0, 0, 0, 1, 1, 1];
        let b = vec![0, 0, 1, 1, 2, 2];
        let (nmi, ami, ari) = gt_metrics(&a, &b);
        assert!((nmi - 0.5158037429793889).abs() < 1e-9, "nmi={nmi}");
        assert!((ami - 0.2987924581708901).abs() < 1e-9, "ami={ami}");
        assert!((ari - 0.24242424242424243).abs() < 1e-9, "ari={ari}");
    }
}
