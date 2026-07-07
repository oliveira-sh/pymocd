//! stats — Welch's two-sample t-test, for the adaptive (plateau) stopping rule.
//! Two-sided p-value via the regularized incomplete beta function (the
//! t-distribution tail), Numerical-Recipes style, reusing `gammln`.

use crate::core::utils::special::gammln;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_var(v: &[f64], m: f64) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
}

/// Continued fraction for the incomplete beta function (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-9 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a, b).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (gammln(a + b) - gammln(a) - gammln(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Welch's two-sample t-test, two-sided p-value that the means of `a` and `b`
/// differ. Returns 1.0 (no evidence of a difference) for degenerate inputs.
pub fn welch_p(a: &[f64], b: &[f64]) -> f64 {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    if n1 < 2.0 || n2 < 2.0 {
        return 1.0;
    }
    let (m1, m2) = (mean(a), mean(b));
    let (v1, v2) = (sample_var(a, m1), sample_var(b, m2));
    let se2 = v1 / n1 + v2 / n2;
    if se2 <= 0.0 {
        return 1.0;
    }
    let t = (m1 - m2) / se2.sqrt();
    let df = se2 * se2 / ((v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0));
    betai(0.5 * df, 0.5, df / (df + t * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_high_p() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(welch_p(&a, &a) > 0.9, "p={}", welch_p(&a, &a));
    }

    #[test]
    fn well_separated_low_p() {
        let a = [0.0, 0.1, -0.1, 0.05, -0.05];
        let b = [10.0, 10.1, 9.9, 10.05, 9.95];
        assert!(welch_p(&a, &b) < 0.001, "p={}", welch_p(&a, &b));
    }

    #[test]
    fn degenerate_returns_one() {
        assert_eq!(welch_p(&[1.0], &[2.0, 3.0]), 1.0);
    }
}
