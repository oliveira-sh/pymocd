//! stats.rs — minimal Welch two-sample t-test p-value for the adaptive stop.
//!
//! Returns the two-sided tail of Student's t with Welch–Satterthwaite degrees of
//! freedom, evaluated through the regularized incomplete beta function (a standard
//! continued-fraction expansion, no external numerical dependency).
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::utils::special::gammln;

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn var(x: &[f64], m: f64) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Two-sided p-value of the Welch (unequal-variance) t-test between samples
/// `a` and `b`. Returns 1.0 (no evidence of a difference) on degenerate input.
pub fn welch_p(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 1.0;
    }
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let se2 = va / na + vb / nb;
    if se2 <= 0.0 {
        return 1.0;
    }
    let t = (ma - mb) / se2.sqrt();
    let df = se2 * se2 / ((va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0));
    if !df.is_finite() || df <= 0.0 {
        return 1.0;
    }
    // two-sided tail: p = I_{df/(df+t^2)}(df/2, 1/2)
    betai(df / 2.0, 0.5, df / (df + t * t))
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

/// Continued-fraction evaluation of the incomplete beta (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn identical_samples_not_significant() {
        let a = [1.0, 1.1, 0.9, 1.05, 0.95];
        let b = [1.02, 1.0, 0.98, 1.03, 0.97];
        assert!(welch_p(&a, &b) > 0.05, "near-identical samples should not be significant");
    }

    #[test]
    fn separated_samples_significant() {
        let a = [0.0, 0.1, -0.1, 0.05, -0.05];
        let b = [5.0, 5.1, 4.9, 5.05, 4.95];
        assert!(welch_p(&a, &b) < 0.001, "well-separated samples should be significant");
    }
}
