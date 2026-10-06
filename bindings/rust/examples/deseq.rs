//! PyDESeq2's workflow on RapidGLM from the command line.
//!
//!   cargo run --release --example deseq -- counts.mtx design.tsv contrast.txt results.tsv
//!
//! counts.mtx: Matrix Market, genes x samples; design.tsv: a header of column
//! names, then one numeric row per sample; contrast.txt: one weight per design
//! column (whitespace separated). Writes baseMean, log2FoldChange, lfcSE, stat,
//! pvalue and padj per gene, in the order of the count matrix.

use rapidglm::deseq::{DeseqDataSet, DeseqOptions, DeseqStats, StatsOptions};
use rapidglm::Matrix;
use std::error::Error;
use std::fs;
use std::io::Write;

fn read_counts(path: &str) -> Result<Matrix, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let header: Vec<usize> = lines
        .next()
        .ok_or("empty count file")?
        .split_whitespace()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let (genes, samples) = (header[0], header[1]);
    let mut m = Matrix::filled(samples, genes, 0.0);
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            continue;
        }
        let (g, s, v): (usize, usize, f64) = (f[0].parse()?, f[1].parse()?, f[2].parse()?);
        m.set(s - 1, g - 1, v);
    }
    Ok(m)
}

fn read_design(path: &str) -> Result<Matrix, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let cols = lines.next().ok_or("empty design file")?.split('\t').count();
    let mut values = Vec::new();
    let mut rows = 0;
    for line in lines {
        values.extend(
            line.split('\t')
                .map(|v| v.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?,
        );
        rows += 1;
    }
    Ok(Matrix::new(rows, cols, values)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        return Err(
            "usage: deseq counts.mtx design.tsv contrast.txt results.tsv [details.tsv]".into(),
        );
    }
    let counts = read_counts(&args[1])?;
    let design = read_design(&args[2])?;
    let contrast: Vec<f64> = fs::read_to_string(&args[3])?
        .split_whitespace()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    eprintln!(
        "RapidGLM {} (C API {})",
        rapidglm::version(),
        rapidglm::c_api_version()
    );
    let start = std::time::Instant::now();
    let mut dds = DeseqDataSet::new(counts, design, DeseqOptions::default())?;
    match std::env::var("RAPIDGLM_TREND") {
        // A given parametric trend "a0,a1" (to compare later steps with another run).
        Ok(t) => {
            let c: Vec<f64> = t.split(',').map(|v| v.parse()).collect::<Result<_, _>>()?;
            dds.fit_size_factors()?;
            dds.fit_genewise_dispersions()?;
            dds.set_parametric_trend(c[0], c[1]);
            dds.fit_dispersion_prior();
            dds.fit_map_dispersions()?;
            dds.fit_lfc()?;
            dds.calculate_cooks();
            dds.refit()?;
            dds.compute_cooks_outlier();
        }
        Err(_) => dds.deseq2()?,
    }
    let res = DeseqStats::new(&dds, contrast, StatsOptions::default())?.summary()?;
    eprintln!(
        "{} genes, {} samples in {:.2} s; {} at padj < 0.05; {} refitted, {} Cook's outliers, {} bounded fits",
        dds.n_genes(),
        dds.n_samples(),
        start.elapsed().as_secs_f64(),
        res.padj.iter().filter(|p| **p < 0.05).count(),
        dds.refitted.iter().filter(|r| **r).count(),
        dds.cooks_outlier.iter().filter(|r| **r).count(),
        dds.bounded_fits()
    );
    let mut out = std::io::BufWriter::new(fs::File::create(&args[4])?);
    writeln!(out, "baseMean\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj")?;
    for j in 0..dds.n_genes() {
        writeln!(
            out,
            "{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}",
            res.base_mean[j],
            res.log2_fold_change[j],
            res.lfc_se[j],
            res.stat[j],
            res.pvalue[j],
            res.padj[j]
        )?;
    }
    if let Some(path) = args.get(5) {
        // Per-gene intermediate values, and the run-level ones as comments.
        let mut out = std::io::BufWriter::new(fs::File::create(path)?);
        writeln!(out, "# size_factors {:?}", dds.size_factors)?;
        writeln!(
            out,
            "# trend_coeffs {:?} mean_disp {:e}",
            dds.trend_coeffs, dds.mean_disp
        )?;
        writeln!(
            out,
            "# squared_logres {:e} prior_disp_var {:e}",
            dds.squared_logres, dds.prior_disp_var
        )?;
        writeln!(
            out,
            "genewise\tfitted\tMAP\tdispersions\tlfc\treplaced\trefitted\tcooks_outlier"
        )?;
        for j in 0..dds.n_genes() {
            let lfc: Vec<String> = dds.lfc.row(j).iter().map(|v| format!("{v:e}")).collect();
            writeln!(
                out,
                "{:e}\t{:e}\t{:e}\t{:e}\t{}\t{}\t{}\t{}",
                dds.genewise_dispersions[j],
                dds.fitted_dispersions[j],
                dds.map_dispersions[j],
                dds.dispersions[j],
                lfc.join(","),
                dds.replaced[j] as u8,
                dds.refitted[j] as u8,
                dds.cooks_outlier[j] as u8
            )?;
        }
    }
    Ok(())
}
