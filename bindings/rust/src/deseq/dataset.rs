//! `DeseqDataSet`: size factors, dispersions, log fold changes and Cook's
//! outliers, following `pydeseq2.dds.DeseqDataSet` (PyDESeq2 0.5.2, MIT,
//! Copyright (c) 2022 Owkin).

use super::util::{
    design_groups, mean_absolute_deviation, median, n_or_more_replicates, trimmed_mean,
};
use crate::linalg::Qr;
use crate::special::{f_ppf, trigamma};
use crate::{nb_dispersions, nb_fixed_fits, Error, Matrix, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeFactorMethod {
    /// Median of ratios.
    Ratio,
    /// Median of ratios over positive counts (for sparse data).
    Poscounts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendType {
    /// a0 + a1 / mean, fitted by a gamma GLM with outlier removal.
    Parametric,
    /// A trimmed mean of gene-wise dispersions.
    Mean,
}

/// Settings with PyDESeq2's names and defaults.
#[derive(Debug, Clone)]
pub struct DeseqOptions {
    pub fit_type: TrendType,
    pub size_factors_fit_type: SizeFactorMethod,
    /// Genes (column indices) used for size factors; None: all.
    pub control_genes: Option<Vec<usize>>,
    pub min_mu: f64,
    pub min_disp: f64,
    pub max_disp: f64,
    pub refit_cooks: bool,
    pub min_replicates: usize,
    pub beta_tol: f64,
    /// RapidGLM threads (default: all cores).
    pub threads: Option<usize>,
}

impl Default for DeseqOptions {
    fn default() -> Self {
        DeseqOptions {
            fit_type: TrendType::Parametric,
            size_factors_fit_type: SizeFactorMethod::Ratio,
            control_genes: None,
            min_mu: 0.5,
            min_disp: 1e-8,
            max_disp: 10.0,
            refit_cooks: true,
            min_replicates: 7,
            beta_tol: 1e-8,
            threads: None,
        }
    }
}

/// Counts (samples x genes), a design matrix (samples x columns) and, after
/// [`DeseqDataSet::deseq2`], the fitted values. Per-gene vectors have one value
/// per gene, NaN for genes with only zero counts.
#[derive(Debug, Clone)]
pub struct DeseqDataSet {
    pub counts: Matrix,
    pub design: Matrix,
    pub options: DeseqOptions,
    max_disp: f64,
    threads: i32,

    pub size_factors: Vec<f64>,
    /// Mean normalized count per gene (PyDESeq2's `_normed_means`).
    pub normed_means: Vec<f64>,
    pub non_zero: Vec<bool>,
    pub mom_dispersions: Vec<f64>,
    pub genewise_dispersions: Vec<f64>,
    pub genewise_converged: Vec<bool>,
    /// Parametric trend coefficients (a0, a1), or None for the mean trend.
    pub trend_coeffs: Option<[f64; 2]>,
    pub mean_disp: f64,
    pub fitted_dispersions: Vec<f64>,
    pub squared_logres: f64,
    pub prior_disp_var: f64,
    pub map_dispersions: Vec<f64>,
    pub map_converged: Vec<bool>,
    /// Final dispersions: MAP, or gene-wise for dispersion outliers.
    pub dispersions: Vec<f64>,
    pub outlier_genes: Vec<bool>,
    /// genes x design columns, natural log scale.
    pub lfc: Matrix,
    pub lfc_converged: Vec<bool>,
    /// samples x genes.
    pub cooks: Matrix,
    pub replaceable: Vec<bool>,
    pub replaced: Vec<bool>,
    pub refitted: Vec<bool>,
    pub new_all_zeroes: Vec<bool>,
    replace_cooks: Option<Matrix>,
    mu_hat: Option<Matrix>,
    mu_lfc: Option<(Matrix, Matrix)>,
    bounded: std::cell::Cell<usize>,
    /// Genes whose p-values are dropped as Cook's outliers.
    pub cooks_outlier: Vec<bool>,
}

/// Results of the per-gene steps on a set of genes (PyDESeq2 runs the same
/// steps again on the genes refitted after outlier replacement).
struct Fits {
    mom: Vec<f64>,
    mu_hat: Matrix,
    genewise: Vec<f64>,
    genewise_converged: Vec<bool>,
}

impl DeseqDataSet {
    pub fn new(counts: Matrix, design: Matrix, options: DeseqOptions) -> Result<DeseqDataSet> {
        if counts.rows != design.rows {
            return Err(Error(
                "counts and design need the same number of samples".into(),
            ));
        }
        if counts
            .values
            .iter()
            .any(|v| !(v.is_finite() && *v >= 0.0 && v.fract() == 0.0))
        {
            return Err(Error("counts must be non-negative integers".into()));
        }
        if design.values.iter().any(|v| v.is_nan()) {
            return Err(Error("NaNs are not allowed in the design".into()));
        }
        if !Qr::new(&design).full_rank(1e-10) {
            return Err(Error("the design matrix is not full rank".into()));
        }
        let (n, genes) = (counts.rows, counts.cols);
        let threads = options.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|t| t.get())
                .unwrap_or(1)
        }) as i32;
        let max_disp = options.max_disp.max(n as f64);
        let p = design.cols;
        Ok(DeseqDataSet {
            max_disp,
            threads,
            size_factors: vec![],
            normed_means: vec![f64::NAN; genes],
            non_zero: vec![false; genes],
            mom_dispersions: vec![f64::NAN; genes],
            genewise_dispersions: vec![f64::NAN; genes],
            genewise_converged: vec![false; genes],
            trend_coeffs: None,
            mean_disp: f64::NAN,
            fitted_dispersions: vec![f64::NAN; genes],
            squared_logres: f64::NAN,
            prior_disp_var: f64::NAN,
            map_dispersions: vec![f64::NAN; genes],
            map_converged: vec![false; genes],
            dispersions: vec![f64::NAN; genes],
            outlier_genes: vec![false; genes],
            lfc: Matrix::filled(genes, p, f64::NAN),
            lfc_converged: vec![false; genes],
            cooks: Matrix::filled(n, genes, f64::NAN),
            replaceable: vec![false; n],
            replaced: vec![false; genes],
            refitted: vec![false; genes],
            new_all_zeroes: vec![false; genes],
            replace_cooks: None,
            mu_hat: None,
            mu_lfc: None,
            bounded: std::cell::Cell::new(0),
            cooks_outlier: vec![false; genes],
            counts,
            design,
            options,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.counts.rows
    }

    pub fn n_genes(&self) -> usize {
        self.counts.cols
    }

    /// Number of NB fits (over all steps) that fell back to [`bounded_fit`].
    pub fn bounded_fits(&self) -> usize {
        self.bounded.get()
    }

    /// The whole first part of the pipeline (PyDESeq2's `deseq2()`).
    pub fn deseq2(&mut self) -> Result<()> {
        self.fit_size_factors()?;
        self.fit_genewise_dispersions()?;
        self.fit_dispersion_trend()?;
        self.fit_dispersion_prior();
        self.fit_map_dispersions()?;
        self.fit_lfc()?;
        self.calculate_cooks();
        if self.options.refit_cooks {
            self.refit()?;
        }
        self.compute_cooks_outlier();
        Ok(())
    }

    fn normed_counts(&self) -> Matrix {
        let mut m = self.counts.clone();
        for i in 0..m.rows {
            for j in 0..m.cols {
                m.set(i, j, m.get(i, j) / self.size_factors[i]);
            }
        }
        m
    }

    pub fn fit_size_factors(&mut self) -> Result<()> {
        let (n, genes) = (self.n_samples(), self.n_genes());
        let mut control = vec![self.options.control_genes.is_none(); genes];
        if let Some(c) = &self.options.control_genes {
            for &g in c {
                *control
                    .get_mut(g)
                    .ok_or_else(|| Error("control gene out of range".into()))? = true;
            }
        }
        let size_factors: Vec<f64> = match self.options.size_factors_fit_type {
            SizeFactorMethod::Poscounts => {
                let logmeans: Vec<f64> = (0..genes)
                    .map(|j| {
                        (0..n)
                            .map(|i| {
                                let v = self.counts.get(i, j);
                                if v > 0.0 {
                                    v.ln()
                                } else {
                                    0.0
                                }
                            })
                            .sum::<f64>()
                            / n as f64
                    })
                    .collect();
                let usable: Vec<bool> = (0..genes)
                    .map(|j| control[j] && logmeans[j].is_finite() && logmeans[j] > 0.0)
                    .collect();
                let sf: Vec<f64> = (0..n)
                    .map(|i| {
                        let ratios: Vec<f64> = (0..genes)
                            .filter(|&j| usable[j] && self.counts.get(i, j) > 0.0)
                            .map(|j| self.counts.get(i, j).ln() - logmeans[j])
                            .collect();
                        median(&ratios).exp()
                    })
                    .collect();
                let geometric = (sf.iter().map(|s| s.ln()).sum::<f64>() / n as f64).exp();
                sf.iter().map(|s| s / geometric).collect()
            }
            SizeFactorMethod::Ratio => {
                let every_gene_has_zero =
                    (0..genes).all(|j| (0..n).any(|i| self.counts.get(i, j) == 0.0));
                if every_gene_has_zero {
                    return Err(Error(
                        "every gene contains a zero: use SizeFactorMethod::Poscounts \
                         (PyDESeq2's iterative size factors are not ported)"
                            .into(),
                    ));
                }
                let logmeans: Vec<f64> = (0..genes)
                    .map(|j| (0..n).map(|i| self.counts.get(i, j).ln()).sum::<f64>() / n as f64)
                    .collect();
                (0..n)
                    .map(|i| {
                        let ratios: Vec<f64> = (0..genes)
                            .filter(|&j| control[j] && logmeans[j].is_finite())
                            .map(|j| self.counts.get(i, j).ln() - logmeans[j])
                            .collect();
                        median(&ratios).exp()
                    })
                    .collect()
            }
        };
        self.size_factors = size_factors;
        let normed = self.normed_counts();
        self.normed_means = (0..genes)
            .map(|j| normed.column(j).iter().sum::<f64>() / n as f64)
            .collect();
        Ok(())
    }

    fn non_zero_index(&self) -> Vec<usize> {
        (0..self.n_genes()).filter(|&j| self.non_zero[j]).collect()
    }

    /// Method-of-moments starting values, the first means and gene-wise
    /// dispersions for the given counts (samples x genes, no all-zero gene).
    fn genewise_fits(&self, counts: &Matrix) -> Result<Fits> {
        let (n, genes, p) = (counts.rows, counts.cols, self.design.cols);
        if n == p {
            return Err(Error(
                "the number of samples equals the number of design variables: \
                 there are no replicates to estimate the dispersion"
                    .into(),
            ));
        }
        let sf = &self.size_factors;
        let qr = Qr::new(&self.design);
        // Rough (linear model) and moments dispersions; the smaller of the two.
        let s_mean_inv = sf.iter().map(|s| 1.0 / s).sum::<f64>() / n as f64;
        let mut mom = vec![0.0; genes];
        for j in 0..genes {
            let y: Vec<f64> = (0..n).map(|i| counts.get(i, j) / sf[i]).collect();
            let beta = qr.solve(&y);
            let mut rough = 0.0;
            for (i, yi) in y.iter().enumerate() {
                let fit = self
                    .design
                    .row(i)
                    .iter()
                    .zip(&beta)
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    .max(1.0);
                rough += ((yi - fit).powi(2) - fit) / ((n - p) as f64 * fit * fit);
            }
            let rough = rough.max(0.0);
            let mean = y.iter().sum::<f64>() / n as f64;
            let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let mut moments = (var - s_mean_inv * mean) / (mean * mean);
            if moments.is_nan() {
                moments = 0.0;
            }
            mom[j] = rough
                .min(moments)
                .clamp(self.options.min_disp, self.max_disp);
        }
        // First means: a linear model when there are as many distinct design
        // rows as columns, otherwise NB fits at the starting dispersions.
        let (_, distinct) = design_groups(&self.design);
        let mu_hat = if distinct == p {
            let mut mu = Matrix::filled(n, genes, 0.0);
            for j in 0..genes {
                let y: Vec<f64> = (0..n).map(|i| counts.get(i, j) / sf[i]).collect();
                let beta = qr.solve(&y);
                for i in 0..n {
                    let fit = self
                        .design
                        .row(i)
                        .iter()
                        .zip(&beta)
                        .map(|(x, b)| x * b)
                        .sum::<f64>();
                    mu.set(i, j, (sf[i] * fit).max(self.options.min_mu));
                }
            }
            mu
        } else {
            let f = self.fixed_fits(counts, &mom)?;
            f.fitted
        };
        let (genewise, converged) = nb_dispersions(
            &self.design,
            counts,
            &mu_hat,
            &mom,
            self.options.min_disp,
            self.max_disp,
            0.0,
            true,
            true,
            self.threads,
        )?;
        let genewise = genewise
            .iter()
            .map(|d| d.clamp(self.options.min_disp, self.max_disp))
            .collect();
        Ok(Fits {
            mom,
            mu_hat,
            genewise,
            genewise_converged: converged,
        })
    }

    /// NB fits at fixed dispersions in RapidGLM; genes it does not converge on
    /// (diverging coefficients pushed against the bounds) are refitted by
    /// [`bounded_fit`].
    fn fixed_fits(&self, counts: &Matrix, dispersions: &[f64]) -> Result<crate::FixedFits> {
        let mut fits = nb_fixed_fits(
            &self.design,
            counts,
            &self.size_factors,
            dispersions,
            self.options.beta_tol,
            1000,
            MIN_BETA,
            MAX_BETA,
            self.threads,
        )?;
        let failed: Vec<usize> = (0..counts.cols).filter(|&j| !fits.converged[j]).collect();
        for j in failed {
            let y = counts.column(j);
            if let Some((beta, mu, hat)) = bounded_fit(
                &self.design,
                &y,
                &self.size_factors,
                dispersions[j],
                self.options.beta_tol,
            ) {
                for (c, b) in beta.iter().enumerate() {
                    fits.coefficients.set(j, c, *b);
                }
                for i in 0..counts.rows {
                    fits.fitted.set(i, j, mu[i]);
                    fits.hat.set(i, j, hat[i]);
                }
                fits.converged[j] = true;
                self.bounded.set(self.bounded.get() + 1);
            }
        }
        Ok(fits)
    }

    fn map_fits(
        &self,
        counts: &Matrix,
        mu_hat: &Matrix,
        fitted: &[f64],
    ) -> Result<(Vec<f64>, Vec<bool>)> {
        let (map, converged) = nb_dispersions(
            &self.design,
            counts,
            mu_hat,
            fitted,
            self.options.min_disp,
            self.max_disp,
            self.prior_disp_var,
            true,
            true,
            self.threads,
        )?;
        Ok((
            map.iter()
                .map(|d| d.clamp(self.options.min_disp, self.max_disp))
                .collect(),
            converged,
        ))
    }

    fn is_dispersion_outlier(&self, genewise: f64, fitted: f64) -> bool {
        genewise.ln() > fitted.ln() + 2.0 * self.squared_logres.sqrt()
    }

    fn trend_at(&self, mean: f64) -> f64 {
        match self.trend_coeffs {
            Some([a0, a1]) => a0 + a1 / mean,
            None => self.mean_disp,
        }
    }

    pub fn fit_genewise_dispersions(&mut self) -> Result<()> {
        let (n, genes) = (self.n_samples(), self.n_genes());
        self.non_zero = (0..genes)
            .map(|j| (0..n).any(|i| self.counts.get(i, j) != 0.0))
            .collect();
        let index = self.non_zero_index();
        let counts = self.counts.select_columns(&index);
        let fits = self.genewise_fits(&counts)?;
        for (k, &j) in index.iter().enumerate() {
            self.mom_dispersions[j] = fits.mom[k];
            self.genewise_dispersions[j] = fits.genewise[k];
            self.genewise_converged[j] = fits.genewise_converged[k];
        }
        self.mu_hat = Some(fits.mu_hat);
        Ok(())
    }

    pub fn fit_dispersion_trend(&mut self) -> Result<()> {
        if self.options.fit_type == TrendType::Parametric && self.fit_parametric_trend() {
            return Ok(());
        }
        // Mean trend (also the fallback when the parametric fit fails).
        let kept: Vec<f64> = self
            .genewise_dispersions
            .iter()
            .copied()
            .filter(|d| *d > 10.0 * self.options.min_disp)
            .collect();
        self.mean_disp = trimmed_mean(&kept, 0.001);
        self.trend_coeffs = None;
        self.fitted_dispersions = vec![self.mean_disp; self.n_genes()];
        Ok(())
    }

    /// Sets the parametric trend a0 + a1 / mean instead of fitting it, for
    /// example to reuse another run's trend (call after the gene-wise
    /// dispersions, then continue with [`DeseqDataSet::fit_dispersion_prior`]).
    pub fn set_parametric_trend(&mut self, a0: f64, a1: f64) {
        self.trend_coeffs = Some([a0, a1]);
        self.fitted_dispersions = (0..self.n_genes())
            .map(|j| {
                if self.non_zero[j] {
                    a0 + a1 / self.normed_means[j]
                } else {
                    f64::NAN
                }
            })
            .collect();
    }

    /// The parametric trend; false if it fails (the caller uses the mean).
    fn fit_parametric_trend(&mut self) -> bool {
        let mut kept: Vec<usize> = self
            .non_zero_index()
            .into_iter()
            .filter(|&j| (1.0 / self.normed_means[j]).is_finite())
            .collect();
        let mut old: [f64; 2] = [0.1, 0.1];
        let mut coeffs: [f64; 2] = [1.0, 1.0];
        while coeffs.iter().all(|c| *c > 1e-10)
            && (coeffs[0] / old[0]).abs().ln().powi(2) + (coeffs[1] / old[1]).abs().ln().powi(2)
                >= 1e-6
        {
            old = coeffs;
            let x: Vec<f64> = kept.iter().map(|&j| 1.0 / self.normed_means[j]).collect();
            let y: Vec<f64> = kept.iter().map(|&j| self.genewise_dispersions[j]).collect();
            match gamma_trend(&x, &y) {
                Some(c) if c.iter().all(|v| *v > 1e-10) => coeffs = c,
                _ => return false,
            }
            kept.retain(|&j| {
                let ratio =
                    self.genewise_dispersions[j] / (coeffs[0] + coeffs[1] / self.normed_means[j]);
                !(ratio < 1e-4 || ratio >= 15.0)
            });
        }
        self.trend_coeffs = Some(coeffs);
        self.fitted_dispersions = (0..self.n_genes())
            .map(|j| {
                if self.non_zero[j] {
                    coeffs[0] + coeffs[1] / self.normed_means[j]
                } else {
                    f64::NAN
                }
            })
            .collect();
        true
    }

    pub fn fit_dispersion_prior(&mut self) {
        let (n, p) = (self.n_samples(), self.design.cols);
        let residuals: Vec<f64> = self
            .non_zero_index()
            .into_iter()
            .filter(|&j| self.genewise_dispersions[j] >= 100.0 * self.options.min_disp)
            .map(|j| self.genewise_dispersions[j].ln() - self.fitted_dispersions[j].ln())
            .collect();
        self.squared_logres = mean_absolute_deviation(&residuals).powi(2);
        self.prior_disp_var = (self.squared_logres - trigamma((n - p) as f64 / 2.0)).max(0.25);
    }

    pub fn fit_map_dispersions(&mut self) -> Result<()> {
        let index = self.non_zero_index();
        let counts = self.counts.select_columns(&index);
        let mu_hat = self
            .mu_hat
            .take()
            .ok_or_else(|| Error("fit gene-wise dispersions first".into()))?;
        let fitted: Vec<f64> = index.iter().map(|&j| self.fitted_dispersions[j]).collect();
        let (map, converged) = self.map_fits(&counts, &mu_hat, &fitted)?;
        for (k, &j) in index.iter().enumerate() {
            self.map_dispersions[j] = map[k];
            self.map_converged[j] = converged[k];
        }
        for j in 0..self.n_genes() {
            self.outlier_genes[j] = self
                .is_dispersion_outlier(self.genewise_dispersions[j], self.fitted_dispersions[j]);
            self.dispersions[j] = if self.outlier_genes[j] {
                self.genewise_dispersions[j]
            } else {
                self.map_dispersions[j]
            };
        }
        Ok(())
    }

    pub fn fit_lfc(&mut self) -> Result<()> {
        let index = self.non_zero_index();
        let counts = self.counts.select_columns(&index);
        let dispersions: Vec<f64> = index.iter().map(|&j| self.dispersions[j]).collect();
        let fits = self.fixed_fits(&counts, &dispersions)?;
        for (k, &j) in index.iter().enumerate() {
            for c in 0..self.design.cols {
                self.lfc.set(j, c, fits.coefficients.get(k, c));
            }
            self.lfc_converged[j] = fits.converged[k];
        }
        self.mu_lfc = Some((fits.fitted, fits.hat));
        Ok(())
    }

    /// Cook's distances, with DESeq2's trimmed-moments dispersion as in
    /// PyDESeq2's `robust_method_of_moments_disp`.
    pub fn calculate_cooks(&mut self) {
        let index = self.non_zero_index();
        let (n, p) = (self.n_samples(), self.design.cols as f64);
        let normed = self.normed_counts().select_columns(&index);
        let robust = robust_method_of_moments_disp(&normed, &self.design);
        let (mu, hat) = self.mu_lfc.as_ref().expect("fit LFCs first");
        for (k, &j) in index.iter().enumerate() {
            for i in 0..n {
                let m = mu.get(i, k);
                let h = hat.get(i, k);
                let r2 = (self.counts.get(i, j) - m).powi(2) / (m + robust[k] * m * m);
                // At leverage 1 the count is fitted exactly and the distance is
                // 0 / 0; PyDESeq2 gets rounding noise there (often inf), R DESeq2
                // mostly reports 0. We use 0.
                let d = if h >= 1.0 - 1e-10 {
                    0.0
                } else {
                    r2 / p * h / (1.0 - h).powi(2)
                };
                self.cooks.set(i, j, d);
            }
        }
    }

    fn cooks_cutoff(&self) -> f64 {
        let (n, p) = (self.n_samples() as f64, self.design.cols as f64);
        f_ppf(0.99, p, n - p)
    }

    /// Replaces outlier counts in samples with enough replicates and refits
    /// the genes concerned (PyDESeq2's `refit`).
    pub fn refit(&mut self) -> Result<()> {
        let (n, genes) = (self.n_samples(), self.n_genes());
        self.replaceable = n_or_more_replicates(&self.design, self.options.min_replicates);
        self.replaced = vec![false; genes];
        self.refitted = vec![false; genes];
        if !self.replaceable.iter().any(|r| *r) {
            return Ok(());
        }
        let cutoff = self.cooks_cutoff();
        let flagged = |i: usize, j: usize| self.cooks.get(i, j) > cutoff;
        self.replaced = (0..genes).map(|j| (0..n).any(|i| flagged(i, j))).collect();
        let replaced: Vec<usize> = (0..genes).filter(|&j| self.replaced[j]).collect();
        if replaced.is_empty() {
            return Ok(());
        }
        // Replacement counts: 20%-trimmed mean of normalized counts times the
        // size factor, truncated to an integer.
        let mut counts = self.counts.select_columns(&replaced);
        for (k, &j) in replaced.iter().enumerate() {
            let normed: Vec<f64> = (0..n)
                .map(|i| self.counts.get(i, j) / self.size_factors[i])
                .collect();
            let base = trimmed_mean(&normed, 0.2);
            for i in 0..n {
                if self.replaceable[i] && flagged(i, j) {
                    counts.set(i, k, (base * self.size_factors[i]).trunc());
                }
            }
        }
        // Genes that became all zero are not refitted.
        let mut refit_columns = Vec::new();
        let mut refit_genes = Vec::new();
        for (k, &j) in replaced.iter().enumerate() {
            if (0..n).all(|i| counts.get(i, k) == 0.0) {
                self.new_all_zeroes[j] = true;
                self.normed_means[j] = 0.0;
                for c in 0..self.design.cols {
                    self.lfc.set(j, c, 0.0);
                }
            } else {
                self.refitted[j] = true;
                refit_columns.push(k);
                refit_genes.push(j);
            }
        }
        if refit_genes.is_empty() {
            return Ok(());
        }
        let counts = counts.select_columns(&refit_columns);
        // The same steps on the replaced counts, with the size factors, trend
        // and prior variance of the full data.
        let fits = self.genewise_fits(&counts)?;
        let normed_means: Vec<f64> = (0..counts.cols)
            .map(|k| {
                (0..n)
                    .map(|i| counts.get(i, k) / self.size_factors[i])
                    .sum::<f64>()
                    / n as f64
            })
            .collect();
        let fitted: Vec<f64> = normed_means.iter().map(|m| self.trend_at(*m)).collect();
        let (map, _) = self.map_fits(&counts, &fits.mu_hat, &fitted)?;
        let dispersions: Vec<f64> = (0..counts.cols)
            .map(|k| {
                if self.is_dispersion_outlier(fits.genewise[k], fitted[k]) {
                    fits.genewise[k]
                } else {
                    map[k]
                }
            })
            .collect();
        let lfc = self.fixed_fits(&counts, &dispersions)?;
        for (k, &j) in refit_genes.iter().enumerate() {
            self.normed_means[j] = normed_means[k];
            for c in 0..self.design.cols {
                self.lfc.set(j, c, lfc.coefficients.get(k, c));
            }
            self.genewise_dispersions[j] = fits.genewise[k];
            self.fitted_dispersions[j] = fitted[k];
            self.dispersions[j] = dispersions[k];
        }
        let mut replace_cooks = self.cooks.clone();
        for &j in &refit_genes {
            for i in 0..n {
                if self.replaceable[i] {
                    replace_cooks.set(i, j, 0.0);
                }
            }
        }
        self.replace_cooks = Some(replace_cooks);
        Ok(())
    }

    /// Genes whose p-values are dropped: a Cook's distance above the cutoff in
    /// a sample with 3 or more replicates, unless 3 or more samples have larger
    /// counts than the sample with the largest Cook's distance.
    pub fn compute_cooks_outlier(&mut self) {
        let (n, genes) = (self.n_samples(), self.n_genes());
        let cutoff = self.cooks_cutoff();
        let use_for_max = n_or_more_replicates(&self.design, 3);
        let refitted_any = self.refitted.iter().any(|r| *r);
        let cooks = match (
            &self.replace_cooks,
            self.options.refit_cooks && refitted_any,
        ) {
            (Some(rc), true) => rc,
            _ => &self.cooks,
        };
        self.cooks_outlier = (0..genes)
            .map(|j| {
                if !(0..n).any(|i| use_for_max[i] && cooks.get(i, j) > cutoff) {
                    return false;
                }
                // First sample with the largest Cook's distance (NumPy argmax).
                let mut pos = 0;
                for i in 1..n {
                    if self.cooks.get(i, j) > self.cooks.get(pos, j) {
                        pos = i;
                    }
                }
                let top = self.counts.get(pos, j);
                (0..n).filter(|&i| self.counts.get(i, j) > top).count() < 3
            })
            .collect();
    }
}

const MIN_BETA: f64 = -30.0;
const MAX_BETA: f64 = 30.0;
const RIDGE: f64 = 1e-6;

/// The NB fit at a fixed dispersion `alpha` within [MIN_BETA, MAX_BETA], for
/// genes whose maximum likelihood estimate does not exist (for example a
/// group of samples with no counts). Maximizes the log likelihood with
/// PyDESeq2's 1e-6 ridge, sum y log mu - (y + 1/alpha) log(1 + alpha mu) -
/// 1e-6 |beta|^2 / 2, which is strictly concave, by projected Newton steps
/// with the expected information and step halving, from PyDESeq2's start
/// (least squares on log(y / s + 0.1)). PyDESeq2 runs L-BFGS-B on the same
/// bounds with means floored at `min_mu`, so these genes can differ. Returns
/// (beta, mu, hat values with the ridge, as PyDESeq2 computes them).
fn bounded_fit(
    x: &Matrix,
    y: &[f64],
    sf: &[f64],
    alpha: f64,
    tol: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (n, p) = (x.rows, x.cols);
    let eta = |beta: &[f64], i: usize| x.row(i).iter().zip(beta).map(|(a, b)| a * b).sum::<f64>();
    let objective = |beta: &[f64]| -> f64 {
        let mut l = -0.5 * RIDGE * beta.iter().map(|b| b * b).sum::<f64>();
        for i in 0..n {
            let mu = sf[i] * eta(beta, i).exp();
            l += y[i] * mu.ln() - (y[i] + 1.0 / alpha) * (alpha * mu).ln_1p();
        }
        l
    };
    let information = |beta: &[f64]| -> Vec<f64> {
        let mut m = vec![0.0; p * p];
        for i in 0..n {
            let mu = sf[i] * eta(beta, i).exp();
            let w = mu / (1.0 + alpha * mu);
            let r = x.row(i);
            for a in 0..p {
                for b in 0..p {
                    m[a * p + b] += w * r[a] * r[b];
                }
            }
        }
        for a in 0..p {
            m[a * p + a] += RIDGE;
        }
        m
    };
    let start: Vec<f64> = (0..n).map(|i| (y[i] / sf[i] + 0.1).ln()).collect();
    let mut beta: Vec<f64> = Qr::new(x)
        .solve(&start)
        .iter()
        .map(|b| b.clamp(MIN_BETA, MAX_BETA))
        .collect();
    let mut current = objective(&beta);
    let mut converged = false;
    for _ in 0..1000 {
        let mut gradient: Vec<f64> = beta.iter().map(|b| -RIDGE * b).collect();
        for i in 0..n {
            let mu = sf[i] * eta(&beta, i).exp();
            let r = (y[i] - mu) / (1.0 + alpha * mu);
            for (g, xa) in gradient.iter_mut().zip(x.row(i)) {
                *g += r * xa;
            }
        }
        // Coefficients at a bound whose gradient points outward stay there.
        let free: Vec<usize> = (0..p)
            .filter(|&k| {
                !((beta[k] <= MIN_BETA && gradient[k] < 0.0)
                    || (beta[k] >= MAX_BETA && gradient[k] > 0.0))
            })
            .collect();
        let info = information(&beta);
        let q = free.len();
        let mut sub = vec![0.0; q * q];
        for (a, &fa) in free.iter().enumerate() {
            for (b, &fb) in free.iter().enumerate() {
                sub[a * q + b] = info[fa * p + fb];
            }
        }
        let inv = crate::linalg::inverse(&sub, q)?;
        let mut direction = vec![0.0; p];
        for (a, &fa) in free.iter().enumerate() {
            direction[fa] = free
                .iter()
                .enumerate()
                .map(|(b, &fb)| inv[a * q + b] * gradient[fb])
                .sum();
        }
        let mut t = 1.0;
        let mut next = None;
        for _ in 0..60 {
            let trial: Vec<f64> = (0..p)
                .map(|k| (beta[k] + t * direction[k]).clamp(MIN_BETA, MAX_BETA))
                .collect();
            let value = objective(&trial);
            if value.is_finite() && value >= current {
                next = Some((trial, value));
                break;
            }
            t *= 0.5;
        }
        let Some((trial, value)) = next else {
            converged = true; // no ascent left at machine precision
            break;
        };
        let change = (value - current).abs() / (value.abs() + 0.1);
        beta = trial;
        current = value;
        if change < tol {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }
    let mu: Vec<f64> = (0..n).map(|i| sf[i] * eta(&beta, i).exp()).collect();
    let covariance = crate::linalg::inverse(&information(&beta), p)?;
    let hat = (0..n)
        .map(|i| {
            let r = x.row(i);
            let h: f64 = (0..p)
                .map(|a| {
                    (0..p)
                        .map(|b| r[a] * covariance[a * p + b] * r[b])
                        .sum::<f64>()
                })
                .sum();
            h * mu[i] / (1.0 + alpha * mu[i])
        })
        .collect();
    Some((beta, mu, hat))
}

/// Gamma GLM with identity link for the dispersion trend: minimizes
/// mean(y / mu + log mu) with mu = a0 + a1 x by Fisher scoring with step
/// halving. None if it fails or a coefficient reaches zero.
fn gamma_trend(x: &[f64], y: &[f64]) -> Option<[f64; 2]> {
    let loss = |c: [f64; 2]| -> f64 {
        x.iter()
            .zip(y)
            .map(|(xi, yi)| {
                let mu = c[0] + c[1] * xi;
                yi / mu + mu.ln()
            })
            .sum::<f64>()
            / x.len() as f64
    };
    let mut c = [1.0, 1.0];
    let mut current = loss(c);
    for _ in 0..200 {
        let (mut g0, mut g1, mut i00, mut i01, mut i11) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (xi, yi) in x.iter().zip(y) {
            let mu = c[0] + c[1] * xi;
            let w = 1.0 / (mu * mu);
            g0 += (yi - mu) * w;
            g1 += (yi - mu) * w * xi;
            i00 += w;
            i01 += w * xi;
            i11 += w * xi * xi;
        }
        let det = i00 * i11 - i01 * i01;
        if det.is_nan() || det <= 0.0 {
            return None;
        }
        let step = [(i11 * g0 - i01 * g1) / det, (i00 * g1 - i01 * g0) / det];
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let trial = [c[0] + t * step[0], c[1] + t * step[1]];
            if trial[0] > 0.0 && trial[1] > 0.0 {
                let l = loss(trial);
                if l.is_finite() && l <= current {
                    accepted = Some((trial, l));
                    break;
                }
            }
            t *= 0.5;
        }
        let (trial, l) = accepted?;
        let change = ((trial[0] - c[0]) / c[0])
            .abs()
            .max(((trial[1] - c[1]) / c[1]).abs());
        c = trial;
        current = l;
        if change < 1e-12 {
            return Some(c);
        }
        if c[0] <= 1e-10 || c[1] <= 1e-10 {
            return None;
        }
    }
    Some(c)
}

/// DESeq2's trimmed method-of-moments dispersion for Cook's distances, as
/// PyDESeq2's `robust_method_of_moments_disp` computes it (floored at 0.04).
fn robust_method_of_moments_disp(normed: &Matrix, design: &Matrix) -> Vec<f64> {
    let (n, genes) = (normed.rows, normed.cols);
    let three_or_more = n_or_more_replicates(design, 3);
    let trim_index = |size: usize| {
        let s = size as f64;
        if s >= 23.5 {
            2
        } else if s >= 3.5 {
            1
        } else {
            0
        }
    };
    let trim_ratio = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 8.0];
    let scale = [2.04, 1.86, 1.51];
    let (groups, k) = design_groups(design);
    (0..genes)
        .map(|j| {
            let column = normed.column(j);
            let v = if three_or_more.iter().any(|t| *t) {
                // Max over cells with 3 or more replicates of the scaled trimmed
                // mean of squared deviations from the cell's trimmed mean.
                (0..k)
                    .filter_map(|g| {
                        let cell: Vec<f64> = (0..n)
                            .filter(|&i| groups[i] == g && three_or_more[i])
                            .map(|i| column[i])
                            .collect();
                        if cell.is_empty() {
                            return None;
                        }
                        let t = trim_index(cell.len());
                        let center = trimmed_mean(&cell, trim_ratio[t]);
                        let squares: Vec<f64> = cell.iter().map(|v| (v - center).powi(2)).collect();
                        Some(scale[t] * trimmed_mean(&squares, trim_ratio[t]))
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                let center = trimmed_mean(&column, 0.125);
                let squares: Vec<f64> = column.iter().map(|v| (v - center).powi(2)).collect();
                1.51 * trimmed_mean(&squares, 0.125)
            };
            let m = column.iter().sum::<f64>() / n as f64;
            ((v - m) / (m * m)).max(0.04)
        })
        .collect()
}
