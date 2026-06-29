//! linalg — self-contained diffusion-kernel similarity matrix for MMCoMO.
//!
//! Paper grounding (mmcomo_final.pdf, p.4): "the initial similarity matrix
//! (denoted by SM) ... size n x n, where each element SM_{i,j} is the
//! diffusion kernel similarity [68] between nodes v_i and v_j."
//!
//! [68] = Kondor-Lafferty diffusion kernel:
//!     SM = exp(beta * H),   H = A - D
//! where H is the negative graph Laplacian: H_{ii} = -deg(v_i),
//! H_{ij} = 1 if i~j else 0. SM is symmetric and entrywise positive
//! (matrix exponential of a real symmetric matrix), NOT row-normalised
//! (normalisation happens only later at the Eq.4 membership level).
//!
//! beta is NEVER specified in the paper (absent from Table II and
//! Section IV-A); the caller supplies it.
//!
//! The matrix exponential is computed self-contained (no ndarray/nalgebra)
//! via a symmetric Jacobi eigendecomposition:  H = Q * diag(lambda) * Q^T,
//! hence  exp(beta*H) = Q * diag(exp(beta*lambda)) * Q^T.
//! This is exact for symmetric H and numerically stable (no overflow from
//! repeated squaring of large negative-Laplacian entries).

use super::*;

/// Diffusion-kernel similarity matrix SM = exp(beta * (A - D)).
///
/// Returns an n x n `Sm` (`Vec<Vec<f64>>`), guaranteed symmetric with
/// strictly positive diagonal.
pub fn diffusion_kernel(g: &Graph, beta: f64) -> Sm {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    // Build H = A - D  (negative graph Laplacian).
    // H[i][i] = -deg(i); H[i][j] = 1.0 for each edge i~j.
    // We derive degree directly from the adjacency list so H is consistent
    // with `adj` regardless of how `g.deg` was populated.
    let mut h = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut d = 0.0f64;
        for &j in &g.adj[i] {
            // Off-diagonal adjacency entry. Use += to tolerate multi-edges /
            // duplicate listings (rare, but keeps A well-defined).
            h[i][j] += 1.0;
            d += 1.0;
        }
        h[i][i] -= d;
    }

    // Symmetrise defensively: A should be symmetric, but undirected adjacency
    // lists can occasionally carry asymmetric duplicates. Jacobi requires an
    // exactly symmetric input.
    for i in 0..n {
        for j in (i + 1)..n {
            let s = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = s;
            h[j][i] = s;
        }
    }

    // Eigendecompose H = Q diag(eig) Q^T.
    let (eig, q) = jacobi_eigen(&h);

    // exp(beta*H) = Q diag(exp(beta*lambda)) Q^T.
    let scaled: Vec<f64> = eig.iter().map(|&lam| (beta * lam).exp()).collect();

    // SM[i][j] = sum_k Q[i][k] * scaled[k] * Q[j][k].
    // Compute QS = Q * diag(scaled) first, then SM = QS * Q^T.
    let mut qs = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            qs[i][k] = q[i][k] * scaled[k];
        }
    }

    let mut sm = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0f64;
            let qi = &qs[i];
            let qj = &q[j];
            for k in 0..n {
                acc += qi[k] * qj[k];
            }
            sm[i][j] = acc;
            sm[j][i] = acc;
        }
    }

    sm
}

/// Dense n x n matrix multiply C = A * B (row-major `Vec<Vec<f64>>`).
/// Kept self-contained for callers that need a plain matmul over `Sm`.
#[allow(dead_code)]
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let inner = b.len();
    let cols = b[0].len();
    let mut c = vec![vec![0.0f64; cols]; n];
    for i in 0..n {
        for k in 0..inner {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            let brow = &b[k];
            let crow = &mut c[i];
            for j in 0..cols {
                crow[j] += aik * brow[j];
            }
        }
    }
    c
}

/// Symmetric eigendecomposition via the cyclic Jacobi rotation method.
///
/// Input: a symmetric n x n matrix `a` (row-major). Output: `(eigenvalues,
/// eigenvectors)` where `eigenvectors[i][k]` is the i-th component of the
/// k-th eigenvector (columns are eigenvectors), so that
/// `A = V * diag(eig) * V^T`.
///
/// Self-contained, pure std. Converges quadratically for symmetric matrices.
fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    // Working copy that the rotations diagonalise.
    let mut m = a.to_vec();
    // Accumulated rotations -> eigenvectors (start as identity).
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    if n == 1 {
        return (vec![m[0][0]], v);
    }

    // Cyclic Jacobi sweeps. 100 sweeps is far beyond what symmetric matrices
    // need (typically < 10); we also early-exit on convergence.
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Sum of squares of off-diagonal entries (upper triangle).
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p][q] * m[p][q];
            }
        }
        if off <= 1e-30 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = m[p][p];
                let aqq = m[q][q];
                // Rotation angle: theta = (aqq - app) / (2*apq).
                let theta = (aqq - app) / (2.0 * apq);
                // t = sign(theta) / (|theta| + sqrt(theta^2 + 1)).
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Apply rotation to rows/cols p and q of M.
                // Update diagonal entries.
                m[p][p] = app - t * apq;
                m[q][q] = aqq + t * apq;
                m[p][q] = 0.0;
                m[q][p] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let aip = m[i][p];
                        let aiq = m[i][q];
                        m[i][p] = c * aip - s * aiq;
                        m[p][i] = m[i][p];
                        m[i][q] = s * aip + c * aiq;
                        m[q][i] = m[i][q];
                    }
                }

                // Accumulate the rotation into the eigenvector matrix.
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    // Eigenvalues are the (now ~diagonal) diagonal entries.
    let eig: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a tiny Graph (path 0-1-2 plus edge 2-3) for tests.
    fn toy_graph() -> Graph {
        // edges: 0-1, 1-2, 2-3
        let adj = vec![
            vec![1usize],       // 0
            vec![0usize, 2],    // 1
            vec![1usize, 3],    // 2
            vec![2usize],       // 3
        ];
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2: f64 = deg.iter().sum();
        Graph { n: 4, adj, deg, m2 }
    }

    #[test]
    fn sm_is_symmetric() {
        let g = toy_graph();
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
        let g = toy_graph();
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
        // exp of a real symmetric matrix is entrywise positive when the
        // off-diagonal pattern is connected (Perron-Frobenius on exp).
        let g = toy_graph();
        let sm = diffusion_kernel(&g, 0.1);
        for i in 0..g.n {
            for j in 0..g.n {
                assert!(sm[i][j] > 0.0, "SM[{i}][{j}] = {} not > 0", sm[i][j]);
            }
        }
    }

    #[test]
    fn beta_zero_gives_identity() {
        // exp(0 * H) = I.
        let g = toy_graph();
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
        // Verify Jacobi: A ~= V diag(eig) V^T on a small symmetric matrix.
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
