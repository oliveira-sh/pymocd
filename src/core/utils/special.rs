//! special.rs — numeric special functions shared across subsystems
//! (the SBM/MDL description-length selector).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

/// ln of the binomial coefficient C(n, k) (real arguments), 0 outside 0<=k<=n.
/// Used by the SBM/MDL frontier selector.
pub fn ln_choose(n: f64, k: f64) -> f64 {
    if k < 0.0 || k > n || n < 0.0 {
        return 0.0;
    }
    gammln(n + 1.0) - gammln(k + 1.0) - gammln(n - k + 1.0)
}

/// Log Gamma (Lanczos approximation, the Numerical Recipes `gammln`).
pub fn gammln(xx: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let x = xx;
    let mut y = xx;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}
