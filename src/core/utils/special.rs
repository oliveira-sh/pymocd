//! Numeric special functions shared across subsystems.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// Log Gamma. Lanczos approximation with g = 5, n = 6; coefficients and the
/// `2.5066282746310005` prefactor are verbatim from Numerical Recipes in C
/// (2nd ed.) section 6.1 `gammln`, accurate to about 2e-10 for `xx > 0`.
/// Do not "tidy" the constants — AMI's exact expected-MI sum reads them.
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
    for c in &COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}
