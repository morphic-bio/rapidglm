// End-to-end checks of the bindings on simulated counts: the PyDESeq2
// workflow (rapidglm::deseq) and RapidGLM's own procedure (rapidglm::bulk_de).
// tests/compare_pydeseq2.py compares the workflow with PyDESeq2 itself.

#![allow(clippy::needless_range_loop)]

use rapidglm::deseq::{DeseqDataSet, DeseqOptions, DeseqStats, StatsOptions};
use rapidglm::{bulk_de, BulkOptions, Matrix};

/// xorshift64* with gamma (Marsaglia-Tsang) and Poisson draws: NB counts as a
/// gamma-Poisson mixture.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 && self.uniform().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
    fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda > 50.0 {
            return (lambda + lambda.sqrt() * self.normal()).round().max(0.0);
        }
        let (limit, mut k, mut p) = ((-lambda).exp(), 0.0, 1.0);
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }
    fn nb(&mut self, mu: f64, dispersion: f64) -> f64 {
        let size = 1.0 / dispersion;
        let lambda = self.gamma(size) * mu / size;
        self.poisson(lambda)
    }
}

/// `samples` samples in two conditions (intercept, condition B); every fifth
/// gene three-fold up or down in B. Gene 7 has no counts and gene 11 none in group A.
fn simulate(samples: usize, genes: usize, seed: u64) -> (Matrix, Matrix) {
    let mut rng = Rng(seed);
    let design = Matrix::new(
        samples,
        2,
        (0..samples).flat_map(|i| [1.0, (i % 2) as f64]).collect(),
    )
    .unwrap();
    let size_factors: Vec<f64> = (0..samples)
        .map(|i| 0.7 + 0.6 * (i as f64 / samples as f64))
        .collect();
    let mut counts = Matrix::filled(samples, genes, 0.0);
    for j in 0..genes {
        let base = 5.0 + rng.gamma(1.5) * 60.0;
        let dispersion = 0.05 + 0.5 / base;
        for i in 0..samples {
            // Up in half the affected genes and down in the other half, so
            // that median-of-ratios size factors stay unbiased.
            let effect = match (j % 10, i % 2) {
                (0, 1) => 3.0,
                (5, 1) => 1.0 / 3.0,
                _ => 1.0,
            };
            counts.set(i, j, rng.nb(size_factors[i] * base * effect, dispersion));
        }
    }
    for i in 0..samples {
        counts.set(i, 7, 0.0);
        if i % 2 == 0 {
            counts.set(i, 11, 0.0);
        }
    }
    (counts, design)
}

#[test]
fn workflow_finds_simulated_effects() {
    let (counts, design) = simulate(12, 1000, 1);
    let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default()).unwrap();
    dds.deseq2().unwrap();
    let res = DeseqStats::new(&dds, vec![0.0, 1.0], StatsOptions::default())
        .unwrap()
        .summary()
        .unwrap();

    // All-zero gene: no fit, base mean 0.
    assert!(!dds.non_zero[7]);
    assert_eq!(res.base_mean[7], 0.0);
    assert!(res.pvalue[7].is_nan() && res.padj[7].is_nan());
    // An all-zero group gives a bounded, finite fit.
    assert!(res.log2_fold_change[11].is_finite() && res.log2_fold_change[11] > 10.0);

    let called = |j: usize| res.padj[j] < 0.05;
    let effects: Vec<usize> = (0..1000).filter(|j| j % 5 == 0 && *j != 0).collect();
    let nulls: Vec<usize> = (0..1000)
        .filter(|j| j % 5 != 0 && *j != 7 && *j != 11)
        .collect();
    let power = effects.iter().filter(|&&j| called(j)).count() as f64 / effects.len() as f64;
    let false_calls = nulls.iter().filter(|&&j| called(j)).count() as f64;
    let fdr = false_calls / (0..1000).filter(|&j| called(j)).count() as f64;
    assert!(power > 0.8, "power {power}");
    assert!(fdr < 0.1, "false discovery proportion {fdr}");
    for &j in &effects {
        if called(j) {
            let truth = if j % 10 == 0 {
                3f64.log2()
            } else {
                -(3f64.log2())
            };
            assert!((res.log2_fold_change[j] - truth).abs() < 1.5);
        }
    }
    // The trend and prior are those of a DESeq2-style fit.
    let [a0, a1] = dds.trend_coeffs.expect("parametric trend");
    assert!(a0 > 0.0 && a1 > 0.0);
    assert!(dds.prior_disp_var >= 0.25);
}

#[test]
fn outliers_are_replaced_and_refitted() {
    // 8 samples per group, so samples are replaceable (min_replicates = 7).
    let (mut counts, design) = simulate(16, 300, 2);
    let gene = 3;
    let top = (0..16).map(|i| counts.get(i, gene)).fold(0.0, f64::max);
    counts.set(4, gene, 50.0 * top + 1000.0);
    let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default()).unwrap();
    dds.deseq2().unwrap();
    assert!(dds.replaced[gene] && dds.refitted[gene]);
    assert!(
        !dds.cooks_outlier[gene],
        "a replaced count is not a p-value outlier"
    );

    // Without refitting the gene is flagged instead.
    let (mut counts, design) = simulate(16, 300, 2);
    counts.set(4, gene, 50.0 * top + 1000.0);
    let options = DeseqOptions {
        refit_cooks: false,
        ..DeseqOptions::default()
    };
    let mut dds = DeseqDataSet::new(counts, design, options).unwrap();
    dds.deseq2().unwrap();
    assert!(dds.cooks_outlier[gene]);
    let res = DeseqStats::new(&dds, vec![0.0, 1.0], StatsOptions::default())
        .unwrap()
        .summary()
        .unwrap();
    assert!(res.pvalue[gene].is_nan());
}

#[test]
fn rapidglm_procedure_through_the_c_layer() {
    assert_eq!(rapidglm::c_api_version(), 2);
    assert!(!rapidglm::version().is_empty());
    let (counts, design) = simulate(12, 1000, 1);
    let table = bulk_de(
        &design,
        &counts,
        &BulkOptions {
            threads: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(table.rows(), 1000);
    for column in [
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "pvalue",
        "padj",
        "dispersion",
    ] {
        assert!(table.numeric(column).is_some(), "{column}");
    }
    assert_eq!(table.vector("size_factors").unwrap().len(), 12);
    assert!(table.scalar("prior_variance").is_finite());

    // The two procedures make nearly the same calls on these data.
    let padj = table.numeric("padj").unwrap();
    let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default()).unwrap();
    dds.deseq2().unwrap();
    let res = DeseqStats::new(&dds, vec![0.0, 1.0], StatsOptions::default())
        .unwrap()
        .summary()
        .unwrap();
    let a: Vec<bool> = padj.iter().map(|p| *p < 0.05).collect();
    let b: Vec<bool> = res.padj.iter().map(|p| *p < 0.05).collect();
    let both = a.iter().zip(&b).filter(|(x, y)| **x && **y).count() as f64;
    let either = a.iter().zip(&b).filter(|(x, y)| **x || **y).count() as f64;
    assert!(both / either > 0.95, "Jaccard {}", both / either);

    // Errors from the C layer arrive as Err.
    let bad = Matrix::filled(12, 2, f64::NAN);
    assert!(bulk_de(&bad, &Matrix::filled(12, 3, 1.0), &BulkOptions::default()).is_err());
}
