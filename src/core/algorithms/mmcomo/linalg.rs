//! linalg — diffusion-kernel similarity matrix SM for MMCoMO.
//!
//! Kondor-Lafferty diffusion kernel [68]: SM = exp(beta * H), H = A - D.
//! Computed via symmetric Jacobi eigendecomposition (self-contained, no deps).
//! beta is NEVER specified in the paper; the caller supplies it.

use super::*;

/// Diffusion-kernel similarity matrix SM = exp(beta * (A - D)).
pub fn diffusion_kernel(g: &Graph, beta: f64) -> Sm {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    // H = A - D (negative graph Laplacian). Degree derived from adj so H stays
    // consistent with `adj` regardless of how `g.deg` was populated.
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

    // Symmetrise defensively: Jacobi requires an exactly symmetric input.
    for i in 0..n {
        for j in (i + 1)..n {
            let s = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = s;
            h[j][i] = s;
        }
    }

    let (eig, q) = jacobi_eigen(&h);

    // exp(beta*H) = Q diag(exp(beta*lambda)) Q^T.
    let scaled: Vec<f64> = eig.iter().map(|&lam| (beta * lam).exp()).collect();

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

/// Symmetric eigendecomposition via cyclic Jacobi rotations.
///
/// `a` symmetric n x n (row-major) -> `(eigenvalues, eigenvectors)` with
/// columns of V the eigenvectors, so `A = V * diag(eig) * V^T`.
fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m = a.to_vec();
    // Accumulated rotations -> eigenvectors (start as identity).
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    if n == 1 {
        return (vec![m[0][0]], v);
    }

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
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

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

                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    let eig: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_graph() -> Graph {
        // edges: 0-1, 1-2, 2-3
        let adj = vec![vec![1usize], vec![0usize, 2], vec![1usize, 3], vec![2usize]];
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
