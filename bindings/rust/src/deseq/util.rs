//! Helpers of the PyDESeq2 workflow: robust summaries, Benjamini-Hochberg,
//! lowess, and grouping of samples by design row.

use crate::Matrix;
use std::collections::HashMap;

/// Median (mean of the two middle values for even lengths); NaN when empty.
pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Mean after removing floor(n * trim) values from each end of the sorted data.
pub fn trimmed_mean(x: &[f64], trim: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let cut = (n as f64 * trim).floor() as usize;
    let kept = &s[cut..n - cut];
    kept.iter().sum::<f64>() / kept.len() as f64
}

/// Scaled median absolute deviation, median(|x - median(x)|) / Phi^-1(0.75).
pub fn mean_absolute_deviation(x: &[f64]) -> f64 {
    let center = median(x);
    let deviations: Vec<f64> = x.iter().map(|v| (v - center).abs()).collect();
    median(&deviations) / crate::special::NORMAL_Q75
}

/// Quantile with linear interpolation between order statistics (NumPy's default).
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let (a, b, t) = (sorted[lo], sorted[hi], h - lo as f64);
    // NumPy's interpolation, which works from the nearer end.
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// Benjamini-Hochberg adjusted p-values (no missing values).
pub fn benjamini_hochberg(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut adjusted = vec![0.0; m];
    let mut running = f64::INFINITY;
    for rank in (0..m).rev() {
        let i = order[rank];
        running = running.min(p[i] * m as f64 / (rank + 1) as f64);
        adjusted[i] = running.min(1.0);
    }
    adjusted
}

/// Robust locally weighted regression (lowess) with tricube weights and
/// bisquare robustness iterations.
///
/// Follows `pydeseq2.utils.lowess`, which is adapted from Alexandre Gramfort's
/// lowess (https://gist.github.com/agramfort/850437; BSD 3-clause licence).
pub fn lowess(x: &[f64], y: &[f64], frac: f64, iterations: usize) -> Vec<f64> {
    let n = x.len();
    let r = (frac * n as f64).ceil() as usize;
    let h: Vec<f64> = (0..n)
        .map(|i| {
            let mut d: Vec<f64> = x.iter().map(|v| (v - x[i]).abs()).collect();
            d.sort_by(f64::total_cmp);
            d[r].max(1e-12)
        })
        .collect();
    // w[j][i]: weight of point j in the fit at point i.
    let w: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let mut u = (x[j] - x[i]) / h[i];
                    if u.is_nan() {
                        u = 0.0;
                    }
                    let u = u.abs().clamp(0.0, 1.0);
                    (1.0 - u * u * u).powi(3)
                })
                .collect()
        })
        .collect();
    let mut estimate = vec![0.0; n];
    let mut delta = vec![1.0; n];
    for _ in 0..iterations {
        for i in 0..n {
            let (mut s0, mut s1, mut s2, mut t0, mut t1) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..n {
                let weight = delta[j] * w[j][i];
                s0 += weight;
                s1 += weight * x[j];
                s2 += weight * x[j] * x[j];
                t0 += weight * y[j];
                t1 += weight * y[j] * x[j];
            }
            let beta = crate::linalg::lstsq_sym2(s0, s1, s2, [t0, t1]);
            estimate[i] = beta[0] + beta[1] * x[i];
        }
        let residuals: Vec<f64> = (0..n).map(|i| y[i] - estimate[i]).collect();
        let s = median(&residuals.iter().map(|r| r.abs()).collect::<Vec<_>>());
        for i in 0..n {
            let d = if s == 0.0 {
                (residuals[i].abs() > 0.0) as u8 as f64
            } else {
                (residuals[i] / (6.0 * s)).clamp(-1.0, 1.0)
            };
            delta[i] = (1.0 - d * d).powi(2);
        }
    }
    estimate
}

/// Group index of each sample by its design row (identical rows share a
/// group), and the number of groups.
pub fn design_groups(design: &Matrix) -> (Vec<usize>, usize) {
    let mut ids: HashMap<Vec<u64>, usize> = HashMap::new();
    let groups = (0..design.rows)
        .map(|i| {
            let key: Vec<u64> = design.row(i).iter().map(|v| v.to_bits()).collect();
            let next = ids.len();
            *ids.entry(key).or_insert(next)
        })
        .collect();
    (groups, ids.len())
}

/// Whether each sample's design row occurs at least `min_replicates` times.
pub fn n_or_more_replicates(design: &Matrix, min_replicates: usize) -> Vec<bool> {
    let (groups, k) = design_groups(design);
    let mut sizes = vec![0usize; k];
    for &g in &groups {
        sizes[g] += 1;
    }
    groups.iter().map(|&g| sizes[g] >= min_replicates).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summaries() {
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        assert_eq!(trimmed_mean(&[1.0, 2.0, 3.0, 4.0, 100.0], 0.2), 3.0);
        let q = quantile(&[1.0, 2.0, 4.0], 0.75);
        assert!((q - 3.0).abs() < 1e-15);
        let adj = benjamini_hochberg(&[0.01, 0.04, 0.03, 0.5]);
        // scipy.stats.false_discovery_control([0.01, 0.04, 0.03, 0.5])
        let want = [0.04, 0.05333333333333334, 0.05333333333333334, 0.5];
        for (a, b) in adj.iter().zip(want) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn lowess_reproduces_a_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 + 3.0 * v).collect();
        for (e, t) in lowess(&x, &y, 0.2, 3).iter().zip(&y) {
            assert!((e - t).abs() < 1e-9);
        }
    }
}
