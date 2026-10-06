//! Small dense linear algebra for the design-sized problems of the workflow
//! (the per-gene GLM work runs in the C layer).

use crate::Matrix;

/// Householder QR of an n x p matrix (n >= p), for least squares.
pub struct Qr {
    n: usize,
    p: usize,
    a: Vec<f64>, // column-major: R above the diagonal, Householder vectors below
    tau: Vec<f64>,
    diag: Vec<f64>,
}

impl Qr {
    pub fn new(x: &Matrix) -> Qr {
        let (n, p) = (x.rows, x.cols);
        let mut a = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                a[j * n + i] = x.get(i, j);
            }
        }
        let mut tau = vec![0.0; p];
        let mut diag = vec![0.0; p];
        for k in 0..p.min(n) {
            let norm = (k..n)
                .map(|i| a[k * n + i] * a[k * n + i])
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                continue;
            }
            let alpha = if a[k * n + k] > 0.0 { -norm } else { norm };
            let v0 = a[k * n + k] - alpha;
            a[k * n + k] = v0;
            let vnorm2 = (k..n).map(|i| a[k * n + i] * a[k * n + i]).sum::<f64>();
            tau[k] = 2.0 / vnorm2;
            for j in k + 1..p {
                let dot = (k..n).map(|i| a[k * n + i] * a[j * n + i]).sum::<f64>();
                let s = tau[k] * dot;
                for i in k..n {
                    a[j * n + i] -= s * a[k * n + i];
                }
            }
            diag[k] = alpha;
        }
        Qr { n, p, a, tau, diag }
    }

    /// Whether R has no diagonal element below `tol` times the largest.
    pub fn full_rank(&self, tol: f64) -> bool {
        let big = self.diag.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        self.n >= self.p && big > 0.0 && self.diag.iter().all(|d| d.abs() > tol * big)
    }

    /// Least-squares coefficients for one right-hand side.
    pub fn solve(&self, y: &[f64]) -> Vec<f64> {
        let (n, p) = (self.n, self.p);
        let mut b = y.to_vec();
        for k in 0..p {
            if self.tau[k] == 0.0 {
                continue;
            }
            let dot = (k..n).map(|i| self.a[k * n + i] * b[i]).sum::<f64>();
            let s = self.tau[k] * dot;
            for (i, bi) in b.iter_mut().enumerate().skip(k) {
                *bi -= s * self.a[k * n + i];
            }
        }
        let mut beta = vec![0.0; p];
        for k in (0..p).rev() {
            let mut s = b[k];
            for j in k + 1..p {
                s -= self.a[j * n + k] * beta[j];
            }
            beta[k] = s / self.diag[k];
        }
        beta
    }
}

/// Inverse of a small square matrix (row-major p x p) by Gauss-Jordan
/// elimination with partial pivoting; None if singular.
pub fn inverse(m: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut a = m.to_vec();
    let mut inv = vec![0.0; p * p];
    for i in 0..p {
        inv[i * p + i] = 1.0;
    }
    for c in 0..p {
        let pivot = (c..p).max_by(|&r, &s| a[r * p + c].abs().total_cmp(&a[s * p + c].abs()))?;
        if a[pivot * p + c] == 0.0 || !a[pivot * p + c].is_finite() {
            return None;
        }
        for j in 0..p {
            a.swap(c * p + j, pivot * p + j);
            inv.swap(c * p + j, pivot * p + j);
        }
        let d = a[c * p + c];
        for j in 0..p {
            a[c * p + j] /= d;
            inv[c * p + j] /= d;
        }
        for r in 0..p {
            if r != c {
                let f = a[r * p + c];
                if f != 0.0 {
                    for j in 0..p {
                        a[r * p + j] -= f * a[c * p + j];
                        inv[r * p + j] -= f * inv[c * p + j];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Minimum-norm least-squares solution of a symmetric 2 x 2 system
/// [[a, b], [b, d]] x = y, with singular values below eps * 2 * the largest
/// treated as zero (NumPy's `lstsq` default cut-off).
pub fn lstsq_sym2(a: f64, b: f64, d: f64, y: [f64; 2]) -> [f64; 2] {
    let mean = 0.5 * (a + d);
    let radius = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    let (l1, l2) = (mean + radius, mean - radius);
    // Unit eigenvectors.
    let (v1, v2) = if b != 0.0 {
        let e1 = [l1 - d, b];
        let n1 = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt();
        let e1 = [e1[0] / n1, e1[1] / n1];
        (e1, [-e1[1], e1[0]])
    } else if a >= d {
        ([1.0, 0.0], [0.0, 1.0])
    } else {
        ([0.0, 1.0], [1.0, 0.0])
    };
    let cut = f64::EPSILON * 2.0 * l1.abs().max(l2.abs());
    let mut x = [0.0, 0.0];
    for (l, v) in [(l1, v1), (l2, v2)] {
        if l.abs() > cut {
            let s = (v[0] * y[0] + v[1] * y[1]) / l;
            x[0] += s * v[0];
            x[1] += s * v[1];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_least_squares() {
        let x = Matrix::new(4, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let qr = Qr::new(&x);
        assert!(qr.full_rank(1e-10));
        let b = qr.solve(&[1.0, 3.0, 5.0, 7.0]);
        assert!((b[0] - 1.0).abs() < 1e-12 && (b[1] - 2.0).abs() < 1e-12);
        let dup = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(!Qr::new(&dup).full_rank(1e-10));
    }

    #[test]
    fn inverse_and_lstsq() {
        let inv = inverse(&[4.0, 1.0, 1.0, 3.0], 2).unwrap();
        assert!((inv[0] - 3.0 / 11.0).abs() < 1e-15 && (inv[1] + 1.0 / 11.0).abs() < 1e-15);
        let x = lstsq_sym2(4.0, 1.0, 3.0, [1.0, 2.0]);
        assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-12 && (x[0] + 3.0 * x[1] - 2.0).abs() < 1e-12);
        // Singular: the minimum-norm solution.
        let x = lstsq_sym2(1.0, 1.0, 1.0, [2.0, 2.0]);
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }
}
