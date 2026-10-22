//! PyDESeq2's workflow on the RapidGLM engine.
//!
//! [`DeseqDataSet`] and [`DeseqStats`] follow PyDESeq2 0.5.2 (MIT licence,
//! Copyright (c) 2022 Owkin; Muzellec et al., Bioinformatics 2023): the same
//! steps, defaults and result columns, ported from its source. The per-gene
//! optimizations run in RapidGLM through the C interface. This port uses
//! unfloored NB fits, unlike the Python inference backend from 1.4.0: fixed-dispersion fits
//! (`rapidglm_nb_fixed_fits`) and gene-wise and MAP dispersions
//! (`rapidglm_nb_dispersions`). Differences from PyDESeq2's default inference:
//!
//! - fits are the exact fixed-dispersion MLE (PyDESeq2's IRLS floors means at
//!   `min_mu` during the fit), and dispersions are maximized by Newton steps in
//!   log dispersion rather than L-BFGS-B; genes starting at the dispersion
//!   floor keep it;
//! - the parametric trend's gamma GLM is fitted by Fisher scoring rather than
//!   L-BFGS-B (same optimum);
//! - Cook's distances at samples with leverage 1 (a count fitted exactly, as
//!   in a saturated design cell; singleton cells in additive designs need not have leverage one) are 0 rather than PyDESeq2's
//!   floating-point value of 0 / 0;
//! - genes whose fixed-dispersion MLE does not exist (an all-zero group, for
//!   example) get a bounded fit in [-30, 30] with PyDESeq2's 1e-6 ridge, by
//!   projected Newton steps, where PyDESeq2 runs L-BFGS-B with means floored at
//!   `min_mu`; dispersion estimates RapidGLM does not converge on are reported
//!   (`genewise_converged`, `map_converged`) without a fallback;
//! - designs are given as matrices with column names (no formulas), contrasts
//!   as vectors; the "iterative" size factors and apeGLM shrinkage are not
//!   ported.
//!
//! ```no_run
//! use rapidglm::deseq::{DeseqDataSet, DeseqOptions, DeseqStats, StatsOptions};
//! # fn run(counts: rapidglm::Matrix, design: rapidglm::Matrix) -> rapidglm::Result<()> {
//! let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default())?;
//! dds.deseq2()?;
//! let contrast = vec![0.0, 1.0]; // the second design column
//! let res = DeseqStats::new(&dds, contrast, StatsOptions::default())?.summary()?;
//! println!("{} genes at padj < 0.05", res.padj.iter().filter(|p| **p < 0.05).count());
//! # Ok(()) }
//! ```

mod dataset;
mod stats;
pub(crate) mod util;

pub use dataset::{DeseqDataSet, DeseqOptions, SizeFactorMethod, TrendType};
pub use stats::{AltHypothesis, DeseqResults, DeseqStats, StatsOptions};
