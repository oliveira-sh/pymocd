//! Adaptive normalization of the objective space.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::Individual;

const NEAR_ZERO: f64 = 1e-10;
const MIN_INTERCEPT: f64 = 1e-6;
const ASF_OFF_AXIS_WEIGHT: f64 = 1e-6;

/// Deb & Jain 2014, Alg. 2: `St`'s objectives translated by the ideal point and
/// scaled by the hyperplane intercepts, in `st_indices` order. Both points are
/// recomputed every call; pymoo instead carries them across generations.
pub fn normalize_objectives(
    combined: &[Individual],
    st_indices: &[usize],
    m: usize,
) -> Vec<Vec<f64>> {
    let ideal = ideal_point(combined, st_indices, m);
    let translated: Vec<Vec<f64>> = st_indices
        .iter()
        .map(|&idx| {
            (0..m)
                .map(|j| combined[idx].objectives[j] - ideal[j])
                .collect()
        })
        .collect();

    let extreme = extreme_points(&translated, m);
    let axis_intercepts = intercepts(&translated, &extreme, m);
    translated
        .iter()
        .map(|t| {
            (0..m)
                .map(|j| {
                    let a = if axis_intercepts[j].abs() < NEAR_ZERO {
                        1.0
                    } else {
                        axis_intercepts[j]
                    };
                    t[j] / a
                })
                .collect()
        })
        .collect()
}

fn ideal_point(combined: &[Individual], st_indices: &[usize], m: usize) -> Vec<f64> {
    let mut ideal = vec![f64::INFINITY; m];
    for &idx in st_indices {
        for (j, id) in ideal.iter_mut().enumerate() {
            *id = id.min(combined[idx].objectives[j]);
        }
    }
    ideal
}

/// Index into `translated` of each axis's extreme point: the member minimizing
/// the achievement scalarizing function weighted 1 on that axis.
fn extreme_points(translated: &[Vec<f64>], m: usize) -> Vec<usize> {
    let mut extreme = vec![0usize; m];
    for (j, ext) in extreme.iter_mut().enumerate() {
        let mut best = f64::INFINITY;
        for (i, t) in translated.iter().enumerate() {
            let asf = (0..m)
                .map(|k| t[k] / if k == j { 1.0 } else { ASF_OFF_AXIS_WEIGHT })
                .fold(f64::NEG_INFINITY, f64::max);
            if asf < best {
                best = asf;
                *ext = i;
            }
        }
    }
    extreme
}

/// Hyperplane intercepts `a_j` through the `M` extreme points, from solving
/// `Z·x = 1` for `x = 1/a_j`. Falls back to `max_x f'_j` (then `1.0`) when the
/// system is singular or an intercept is non-positive (Deb & Jain 2014, §IV-C).
fn intercepts(translated: &[Vec<f64>], extreme: &[usize], m: usize) -> Vec<f64> {
    let fallback = || -> Vec<f64> {
        (0..m)
            .map(|j| {
                let mx = translated.iter().map(|t| t[j]).fold(0.0_f64, f64::max);
                if mx > NEAR_ZERO { mx } else { 1.0 }
            })
            .collect()
    };
    let z: Vec<Vec<f64>> = extreme.iter().map(|&i| translated[i].clone()).collect();
    match gaussian_solve(z, vec![1.0; m]) {
        Some(x) if x.iter().all(|&v| v.abs() > NEAR_ZERO) => {
            let a: Vec<f64> = x.iter().map(|&v| 1.0 / v).collect();
            if a.iter().all(|&aj| aj > MIN_INTERCEPT) {
                a
            } else {
                fallback()
            }
        }
        _ => fallback(),
    }
}

/// Gauss-Jordan solve with partial pivoting; `None` if (near-)singular.
fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < NEAR_ZERO {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        let pivot_row = a[col].clone();
        let pivot_b = b[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col] / d;
            for (c, val) in a[r].iter_mut().enumerate().skip(col) {
                *val -= factor * pivot_row[c];
            }
            b[r] -= factor * pivot_b;
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_solve_identity() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let x = gaussian_solve(a, vec![2.0, 4.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9);
        assert!(gaussian_solve(vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![1.0, 1.0]).is_none());
    }
}
