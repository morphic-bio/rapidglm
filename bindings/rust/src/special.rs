//! Special functions and distributions used by the workflow.

/// Standard normal quantile at 0.75 (scales the median absolute deviation).
pub const NORMAL_Q75: f64 = 0.674_489_750_196_081_7;

/// Upper tail of the standard normal distribution.
pub fn normal_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / std::f64::consts::SQRT_2)
}

/// Trigamma function for x > 0: recurrence up to x >= 12, then the asymptotic
/// series.
pub fn trigamma(mut x: f64) -> f64 {
    let mut sum = 0.0;
    while x < 12.0 {
        sum += 1.0 / (x * x);
        x += 1.0;
    }
    let z = 1.0 / (x * x);
    sum + 1.0 / x
        + z / 2.0
        + z / x
            * (1.0 / 6.0 - z * (1.0 / 30.0 - z * (1.0 / 42.0 - z * (1.0 / 30.0 - z * 5.0 / 66.0))))
}

/// Continued fraction for the incomplete beta function (modified Lentz).
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        for numerator in [
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 + numerator * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = 1.0 + numerator / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a, b).
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_front =
        a * x.ln() + b * (-x).ln_1p() - (libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b));
    if x < (a + 1.0) / (a + b + 2.0) {
        log_front.exp() * beta_fraction(a, b, x) / a
    } else {
        1.0 - log_front.exp() * beta_fraction(b, a, 1.0 - x) / b
    }
}

/// Quantile of the F distribution with (d1, d2) degrees of freedom.
pub fn f_ppf(q: f64, d1: f64, d2: f64) -> f64 {
    if d2 <= 0.0 {
        return f64::NAN;
    }
    // F <= f  <=>  y = d1 f / (d1 f + d2) <= y(f); bisect on y in (0, 1).
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_inc(d1 / 2.0, d2 / 2.0, mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= f64::EPSILON * hi {
            break;
        }
    }
    let y = 0.5 * (lo + hi);
    d2 * y / (d1 * (1.0 - y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        // Reference values from SciPy (scipy.stats / scipy.special).
        assert!((normal_sf(1.96) - 0.024997895148220435).abs() < 1e-15);
        assert!((normal_sf(-1.0) - 0.8413447460685429).abs() < 1e-15);
        assert!((normal_sf(9.0) - 1.1285884059538324e-19).abs() < 1e-30);
        assert!((trigamma(0.5) - 4.93480220054468).abs() < 1e-12);
        assert!((trigamma(1.5) - 0.9348022005446793).abs() < 1e-12);
        assert!((trigamma(12.0) - 0.0869018728717684).abs() < 1e-13);
        assert!((f_ppf(0.99, 2.0, 6.0) - 10.92476650083833).abs() < 1e-9);
        assert!((f_ppf(0.99, 11.0, 636.0) - 2.2756515216189426).abs() < 1e-9);
        assert!((f_ppf(0.99, 4.0, 4.0) - 15.977024852557667).abs() < 1e-9);
    }
}
