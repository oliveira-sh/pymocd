//! Diffusion-kernel similarity matrix SM and the eigensolver it needs.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

// dense matrix kernels: indexed loops over several matrices read clearer here
#![allow(clippy::needless_range_loop)]

use crate::core::algorithms::mmcomo::{Graph, Sm};

/// Cyclic Jacobi sweep budget; convergence normally breaks out far earlier.
const MAX_JACOBI_SWEEPS: usize = 100;
/// Sum of squared off-diagonals below which the matrix counts as diagonal.
const OFF_DIAG_TOL: f64 = 1e-30;
/// A pivot this small rotates nothing and would only risk overflowing `theta`.
const MIN_PIVOT: f64 = 1e-300;

/// Kondor-Lafferty diffusion kernel [68]: `SM = exp(beta * (A - D))`, computed
/// via symmetric Jacobi eigendecomposition (self-contained, no deps).
///
/// `beta` is NEVER specified in the paper; the caller supplies it.
pub fn diffusion_kernel(g: &Graph, beta: f64) -> Sm {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    let h = negative_laplacian(g);
    let (eigenvalues, eigenvectors) = jacobi_eigen(&h);

    // exp(beta*H) = Q diag(exp(beta*lambda)) Q^T.
    let scaled: Vec<f64> = eigenvalues.iter().map(|&lam| (beta * lam).exp()).collect();

    let mut q_scaled = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            q_scaled[i][k] = eigenvectors[i][k] * scaled[k];
        }
    }

    let mut sm = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0f64;
            let qi = &q_scaled[i];
            let qj = &eigenvectors[j];
            for k in 0..n {
                acc += qi[k] * qj[k];
            }
            sm[i][j] = acc;
            sm[j][i] = acc;
        }
    }

    sm
}

/// `H = A - D`, exactly symmetric so Jacobi accepts it. Degrees are derived from
/// `adj` rather than read from `g.deg`, so H stays consistent with `adj`
/// regardless of how `g.deg` was populated.
fn negative_laplacian(g: &Graph) -> Vec<Vec<f64>> {
    let n = g.n;
    let mut h = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut d = 0.0f64;
        for &j in &g.adj[i] {
            // += to tolerate multi-edges / duplicate listings.
            h[i][j] += 1.0;
            d += 1.0;
        }
        h[i][i] -= d;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let s = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = s;
            h[j][i] = s;
        }
    }
    h
}

/// Cyclic Jacobi eigendecomposition of a symmetric row-major `a`, returning
/// `(eigenvalues, V)` with the eigenvectors as columns of `V`: `a = V diag(eig) V^T`.
fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut work = a.to_vec();
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    if n == 1 {
        return (vec![work[0][0]], v);
    }

    for _ in 0..MAX_JACOBI_SWEEPS {
        let mut off_diag_sq = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_sq += work[p][q] * work[p][q];
            }
        }
        if off_diag_sq <= OFF_DIAG_TOL {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[p][q];
                if apq.abs() <= MIN_PIVOT {
                    continue;
                }
                let app = work[p][p];
                let aqq = work[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                work[p][p] = app - t * apq;
                work[q][q] = aqq + t * apq;
                work[p][q] = 0.0;
                work[q][p] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let aip = work[i][p];
                        let aiq = work[i][q];
                        work[i][p] = c * aip - s * aiq;
                        work[p][i] = work[i][p];
                        work[i][q] = s * aip + c * aiq;
                        work[q][i] = work[i][q];
                    }
                }

                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    let eig: Vec<f64> = (0..n).map(|i| work[i][i]).collect();
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path 0-1-2-3.
    fn path_of_four() -> Graph {
        let adj = vec![vec![1usize], vec![0usize, 2], vec![1usize, 3], vec![2usize]];
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2: f64 = deg.iter().sum();
        Graph { n: 4, adj, deg, m2 }
    }

    #[test]
    fn sm_is_symmetric() {
        let g = path_of_four();
        let sm = diffusion_kernel(&g, 0.05);
        let n = g.n;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (sm[i][j] - sm[j][i]).abs() < 1e-9,
                    "SM not symmetric at ({i},{j}): {} vs {}",
                    sm[i][j],
                    sm[j][i]
                );
            }
        }
    }

    #[test]
    fn sm_positive_diagonal() {
        let g = path_of_four();
        let sm = diffusion_kernel(&g, 0.05);
        for i in 0..g.n {
            assert!(
                sm[i][i] > 0.0,
                "diagonal entry {i} not positive: {}",
                sm[i][i]
            );
        }
    }

    #[test]
    fn sm_entrywise_positive() {
        let g = path_of_four();
        let sm = diffusion_kernel(&g, 0.1);
        for i in 0..g.n {
            for j in 0..g.n {
                assert!(sm[i][j] > 0.0, "SM[{i}][{j}] = {} not > 0", sm[i][j]);
            }
        }
    }

    #[test]
    fn beta_zero_gives_identity() {
        let g = path_of_four();
        let sm = diffusion_kernel(&g, 0.0);
        for i in 0..g.n {
            for j in 0..g.n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sm[i][j] - expected).abs() < 1e-9,
                    "exp(0*H)[{i}][{j}] = {} expected {expected}",
                    sm[i][j]
                );
            }
        }
    }

    #[test]
    fn eigen_reconstructs_matrix() {
        let a = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let (eig, v) = jacobi_eigen(&a);
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += v[i][k] * eig[k] * v[j][k];
                }
                assert!(
                    (acc - a[i][j]).abs() < 1e-9,
                    "reconstruction mismatch at ({i},{j}): {acc} vs {}",
                    a[i][j]
                );
            }
        }
    }
}
