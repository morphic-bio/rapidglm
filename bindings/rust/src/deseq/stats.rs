//! `DeseqStats`: Wald tests, Cook's filtering and independent filtering,
//! following `pydeseq2.ds.DeseqStats` (PyDESeq2 0.5.2, MIT, Copyright (c) 2022
//! Owkin). LFC shrinkage (apeGLM in PyDESeq2) is not ported.

use super::util::{benjamini_hochberg, lowess, quantile};
use super::DeseqDataSet;
use crate::linalg::inverse;
use crate::special::normal_sf;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AltHypothesis {
    GreaterAbs,
    LessAbs,
    Greater,
    Less,
}

#[derive(Debug, Clone)]
pub struct StatsOptions {
    pub alpha: f64,
    pub cooks_filter: bool,
    pub independent_filter: bool,
    /// Prior standard deviations of the coefficients for the Wald ridge
    /// (PyDESeq2's `prior_LFC_var`); None: a 1e-6 ridge.
    pub prior_lfc_var: Option<Vec<f64>>,
    /// Log2 fold change under the null hypothesis.
    pub lfc_null: f64,
    pub alt_hypothesis: Option<AltHypothesis>,
}

impl Default for StatsOptions {
    fn default() -> Self {
        StatsOptions {
            alpha: 0.05,
            cooks_filter: true,
            independent_filter: true,
            prior_lfc_var: None,
            lfc_null: 0.0,
            alt_hypothesis: None,
        }
    }
}

/// PyDESeq2's `results_df`: one value per gene.
#[derive(Debug, Clone)]
pub struct DeseqResults {
    pub base_mean: Vec<f64>,
    pub log2_fold_change: Vec<f64>,
    pub lfc_se: Vec<f64>,
    pub stat: Vec<f64>,
    pub pvalue: Vec<f64>,
    pub padj: Vec<f64>,
}

pub struct DeseqStats<'a> {
    dds: &'a DeseqDataSet,
    contrast: Vec<f64>,
    options: StatsOptions,
}

impl<'a> DeseqStats<'a> {
    /// `contrast` has one weight per design column.
    pub fn new(dds: &'a DeseqDataSet, contrast: Vec<f64>, options: StatsOptions) -> Result<Self> {
        if dds.size_factors.is_empty() || dds.lfc_converged.iter().all(|c| !c) {
            return Err(Error("run DeseqDataSet::deseq2 first".into()));
        }
        if contrast.len() != dds.design.cols {
            return Err(Error(
                "the contrast vector must have the same length as the design matrix".into(),
            ));
        }
        if options.lfc_null < 0.0
            && matches!(
                options.alt_hypothesis,
                Some(AltHypothesis::GreaterAbs | AltHypothesis::LessAbs)
            )
        {
            return Err(Error(
                "greaterAbs and lessAbs need a non-negative lfc_null".into(),
            ));
        }
        Ok(DeseqStats {
            dds,
            contrast,
            options,
        })
    }

    pub fn summary(&self) -> Result<DeseqResults> {
        let dds = self.dds;
        let genes = dds.n_genes();
        let (mut pvalue, mut stat, mut se) = self.wald_test()?;
        if dds.options.refit_cooks && dds.replaced.iter().any(|r| *r) {
            for j in (0..genes).filter(|&j| dds.new_all_zeroes[j]) {
                se[j] = 0.0;
                stat[j] = 0.0;
                pvalue[j] = 1.0;
            }
        }
        if self.options.cooks_filter {
            for j in (0..genes).filter(|&j| dds.cooks_outlier[j]) {
                pvalue[j] = f64::NAN;
            }
        }
        let padj = if self.options.independent_filter {
            self.independent_filtering(&pvalue)
        } else {
            let tested: Vec<usize> = (0..genes).filter(|&j| !pvalue[j].is_nan()).collect();
            let adjusted =
                benjamini_hochberg(&tested.iter().map(|&j| pvalue[j]).collect::<Vec<_>>());
            let mut padj = vec![f64::NAN; genes];
            for (k, &j) in tested.iter().enumerate() {
                padj[j] = adjusted[k];
            }
            padj
        };
        let ln2 = std::f64::consts::LN_2;
        let log2_fold_change = (0..genes)
            .map(|j| {
                dds.lfc
                    .row(j)
                    .iter()
                    .zip(&self.contrast)
                    .map(|(b, c)| b * c)
                    .sum::<f64>()
                    / ln2
            })
            .collect();
        Ok(DeseqResults {
            base_mean: dds.normed_means.clone(),
            log2_fold_change,
            lfc_se: se.iter().map(|s| s / ln2).collect(),
            stat,
            pvalue,
            padj,
        })
    }

    /// Per gene: covariance H M H with M = X'WX, W = mu / (1 + disp mu),
    /// H = (M + ridge)^-1, at mu = s exp(X beta). Returns (p, statistic, SE).
    fn wald_test(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let dds = self.dds;
        let (n, p, genes) = (dds.n_samples(), dds.design.cols, dds.n_genes());
        let ridge: Vec<f64> = match &self.options.prior_lfc_var {
            Some(v) if v.len() == p => v.iter().map(|s| 1.0 / (s * s)).collect(),
            Some(_) => {
                return Err(Error(
                    "prior_lfc_var needs one value per design column".into(),
                ))
            }
            None => vec![1e-6; p],
        };
        let lfc_null = std::f64::consts::LN_2 * self.options.lfc_null;
        let c = &self.contrast;
        let (mut pvalue, mut stat, mut se) = (
            vec![f64::NAN; genes],
            vec![f64::NAN; genes],
            vec![f64::NAN; genes],
        );
        let mut m = vec![0.0; p * p];
        for j in 0..genes {
            let beta = dds.lfc.row(j);
            let disp = dds.dispersions[j];
            if beta.iter().any(|b| b.is_nan()) || disp.is_nan() {
                continue;
            }
            m.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..n {
                let x = dds.design.row(i);
                let mu =
                    dds.size_factors[i] * x.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>().exp();
                let w = mu / (1.0 + mu * disp);
                for a in 0..p {
                    for b in 0..p {
                        m[a * p + b] += w * x[a] * x[b];
                    }
                }
            }
            let mut regularized = m.clone();
            for a in 0..p {
                regularized[a * p + a] += ridge[a];
            }
            let Some(h) = inverse(&regularized, p) else {
                continue;
            };
            let hc: Vec<f64> = (0..p)
                .map(|a| (0..p).map(|b| h[a * p + b] * c[b]).sum())
                .collect();
            let variance: f64 = (0..p)
                .map(|a| hc[a] * (0..p).map(|b| m[a * p + b] * hc[b]).sum::<f64>())
                .sum();
            let s = variance.sqrt();
            let (t, pv) = self.statistic(beta, lfc_null, s);
            pvalue[j] = pv;
            stat[j] = t;
            se[j] = s;
        }
        Ok((pvalue, stat, se))
    }

    fn statistic(&self, lfc: &[f64], lfc_null: f64, se: f64) -> (f64, f64) {
        let c = &self.contrast;
        let dot = |f: &dyn Fn(f64) -> f64| lfc.iter().zip(c).map(|(b, w)| w * f(*b)).sum::<f64>();
        let greater = |null: f64| {
            let t = dot(&|b| ((b - null) / se).max(0.0));
            (t, normal_sf(t))
        };
        let less = |null: f64| {
            let t = dot(&|b| ((b - null) / se).min(0.0));
            (t, normal_sf(t.abs()))
        };
        match self.options.alt_hypothesis {
            None => {
                let t = dot(&|b| b - lfc_null) / se;
                (t, 2.0 * normal_sf(t.abs()))
            }
            Some(AltHypothesis::Greater) => greater(lfc_null),
            Some(AltHypothesis::Less) => less(lfc_null),
            Some(AltHypothesis::GreaterAbs) => {
                let t = dot(&|b| b.signum() * ((b.abs() - lfc_null) / se).max(0.0));
                (t, 2.0 * normal_sf(t.abs()))
            }
            Some(AltHypothesis::LessAbs) => {
                let (t_above, p_above) = greater(-lfc_null.abs());
                let (t_below, p_below) = less(lfc_null.abs());
                let t = if t_below.abs() < t_above.abs() {
                    t_below
                } else {
                    t_above
                };
                (t, p_above.max(p_below))
            }
        }
    }

    /// Benjamini-Hochberg after filtering on the mean of normalized counts,
    /// at the cut chosen from a lowess fit of rejections against the filtered
    /// fraction (PyDESeq2's `_independent_filtering`).
    fn independent_filtering(&self, pvalue: &[f64]) -> Vec<f64> {
        let base_mean = &self.dds.normed_means;
        let genes = base_mean.len();
        let lower = base_mean.iter().filter(|m| **m == 0.0).count() as f64 / genes as f64;
        let upper = if lower < 0.95 { 0.95 } else { 1.0 };
        // As numpy.linspace: start + k * step, with the last point exactly `upper`.
        let step = (upper - lower) / 49.0;
        let mut theta: Vec<f64> = (0..50).map(|k| lower + k as f64 * step).collect();
        theta[49] = upper;
        let mut sorted = base_mean.clone();
        sorted.sort_by(f64::total_cmp);
        let results: Vec<Vec<f64>> = theta
            .iter()
            .map(|&t| {
                let cutoff = quantile(&sorted, t);
                let used: Vec<usize> = (0..genes)
                    .filter(|&j| base_mean[j] >= cutoff && !pvalue[j].is_nan())
                    .collect();
                let adjusted =
                    benjamini_hochberg(&used.iter().map(|&j| pvalue[j]).collect::<Vec<_>>());
                let mut column = vec![f64::NAN; genes];
                for (k, &j) in used.iter().enumerate() {
                    column[j] = adjusted[k];
                }
                column
            })
            .collect();
        let rejections: Vec<f64> = results
            .iter()
            .map(|column| column.iter().filter(|p| **p < self.options.alpha).count() as f64)
            .collect();
        let smooth = lowess(&theta, &rejections, 1.0 / 5.0, 3);
        let max_rejections = rejections.iter().fold(0.0f64, |a, b| a.max(*b));
        let j = if max_rejections <= 10.0 {
            0
        } else {
            let residuals: Vec<f64> = (0..theta.len())
                .filter(|&k| rejections[k] > 0.0)
                .map(|k| rejections[k] - smooth[k])
                .collect();
            let rms =
                (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
            let threshold = smooth.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)) - rms;
            rejections.iter().position(|r| *r > threshold).unwrap_or(0)
        };
        results[j].clone()
    }
}
