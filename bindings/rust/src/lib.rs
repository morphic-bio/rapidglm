//! Rust bindings to RapidGLM through its C interface (`librapidglm_c`; see
//! docs/C_API.md), and [`deseq`], PyDESeq2's workflow on the RapidGLM engine.
//!
//! Matrices are observations (samples) by features (genes), row-major, as in the
//! C interface. Build RapidGLM first and set `RAPIDGLM_LIB_DIR` to the directory
//! holding `librapidglm_c.so`.

// Numerical loops index several arrays at once.
#![allow(clippy::needless_range_loop)]

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_int;

pub mod deseq;
pub mod ffi;
mod linalg;
mod special;

/// An error reported by the C layer, or invalid input caught on the Rust side.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(status: c_int) -> Result<()> {
    if status == ffi::RAPIDGLM_OK {
        return Ok(());
    }
    // SAFETY: the C layer returns a valid, NUL-terminated per-thread message.
    let message = unsafe { CStr::from_ptr(ffi::rapidglm_last_error()) };
    Err(Error(message.to_string_lossy().into_owned()))
}

/// The library version string, e.g. "0.2.0".
pub fn version() -> String {
    // SAFETY: a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::rapidglm_version()) }
        .to_string_lossy()
        .into_owned()
}

/// The C API version the library implements (this crate is written for 1).
pub fn c_api_version() -> i32 {
    unsafe { ffi::rapidglm_c_api_version() }
}

/// A dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f64>) -> Result<Matrix> {
        if values.len() != rows * cols {
            return Err(Error(format!(
                "matrix of {rows} x {cols} given {} values",
                values.len()
            )));
        }
        Ok(Matrix { rows, cols, values })
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Matrix {
        Matrix {
            rows,
            cols,
            values: vec![value; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.cols + j] = value;
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.cols..(i + 1) * self.cols]
    }

    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// The listed columns, in order.
    pub fn select_columns(&self, columns: &[usize]) -> Matrix {
        let mut values = Vec::with_capacity(self.rows * columns.len());
        for i in 0..self.rows {
            values.extend(columns.iter().map(|&j| self.get(i, j)));
        }
        Matrix {
            rows: self.rows,
            cols: columns.len(),
            values,
        }
    }

    fn view(&self) -> ffi::rapidglm_dense {
        ffi::rapidglm_dense {
            rows: self.rows,
            cols: self.cols,
            values: self.values.as_ptr(),
        }
    }
}

/// A result table from the C layer; freed on drop.
pub struct Table {
    raw: *mut ffi::rapidglm_table,
}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: `raw` came from the C layer and is freed exactly once.
        unsafe { ffi::rapidglm_table_free(self.raw) }
    }
}

impl Table {
    pub fn rows(&self) -> usize {
        unsafe { ffi::rapidglm_table_rows(self.raw) }
    }

    pub fn column_names(&self) -> Vec<String> {
        let n = unsafe { ffi::rapidglm_table_columns(self.raw) };
        (0..n)
            .map(|j| {
                unsafe { CStr::from_ptr(ffi::rapidglm_table_column_name(self.raw, j)) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    /// A numeric column, or None if there is no such column.
    pub fn numeric(&self, name: &str) -> Option<Vec<f64>> {
        let name = CString::new(name).ok()?;
        let p = unsafe { ffi::rapidglm_table_numeric(self.raw, name.as_ptr()) };
        // SAFETY: a non-null pointer refers to `rows` doubles owned by the table.
        (!p.is_null()).then(|| unsafe { std::slice::from_raw_parts(p, self.rows()) }.to_vec())
    }

    /// A text column, or None if there is no such column.
    pub fn text(&self, name: &str) -> Option<Vec<String>> {
        let name = CString::new(name).ok()?;
        let mut out = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            let p = unsafe { ffi::rapidglm_table_text(self.raw, name.as_ptr(), i) };
            if p.is_null() {
                return None;
            }
            out.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
        }
        Some(out)
    }

    /// A run-level scalar (NaN if absent).
    pub fn scalar(&self, name: &str) -> f64 {
        let name = CString::new(name).expect("name without NUL");
        unsafe { ffi::rapidglm_table_scalar(self.raw, name.as_ptr()) }
    }

    /// A run-level vector, or None if absent.
    pub fn vector(&self, name: &str) -> Option<Vec<f64>> {
        let name = CString::new(name).ok()?;
        let mut length = 0usize;
        let p = unsafe { ffi::rapidglm_table_vector(self.raw, name.as_ptr(), &mut length) };
        (!p.is_null()).then(|| unsafe { std::slice::from_raw_parts(p, length) }.to_vec())
    }
}

/// Options of RapidGLM's own DESeq2-procedure analysis ([`bulk_de`]); the
/// fields not listed keep the C layer's defaults (docs/CLI.md).
#[derive(Debug, Clone, Default)]
pub struct BulkOptions {
    /// Tested design column (default: the last).
    pub coefficient: Option<usize>,
    /// A contrast over the design columns instead of one coefficient.
    pub contrast: Option<Vec<f64>>,
    pub alpha: Option<f64>,
    pub threads: Option<i32>,
    /// DESeq2's Cook's-distance dispersion as in PyDESeq2.
    pub legacy_cooks: bool,
}

/// RapidGLM's bulk differential expression (DESeq2 procedure), as in the CLI's
/// `bulk-de`. `counts` is samples x genes. Columns: baseMean, log2FoldChange,
/// lfcSE, stat, pvalue, padj, dispersions and Cook's values (rapidglm_c.h).
pub fn bulk_de(design: &Matrix, counts: &Matrix, options: &BulkOptions) -> Result<Table> {
    if design.rows != counts.rows {
        return Err(Error(
            "design and counts need the same number of samples".into(),
        ));
    }
    // Counts in compressed sparse column form, genes as columns.
    let mut ptr = vec![0usize];
    let (mut index, mut values) = (Vec::new(), Vec::new());
    for j in 0..counts.cols {
        for i in 0..counts.rows {
            let v = counts.get(i, j);
            if v != 0.0 {
                index.push(i);
                values.push(v);
            }
        }
        ptr.push(index.len());
    }
    let sparse = ffi::rapidglm_sparse {
        rows: counts.rows,
        cols: counts.cols,
        column_ptr: ptr.as_ptr(),
        row_index: index.as_ptr(),
        values: values.as_ptr(),
    };
    let mut o = std::mem::MaybeUninit::<ffi::rapidglm_bulk_options>::zeroed();
    // SAFETY: init fills every field, including struct_size.
    let mut o = unsafe {
        ffi::rapidglm_bulk_options_init(o.as_mut_ptr());
        o.assume_init()
    };
    if let Some(c) = options.coefficient {
        o.coefficient = c;
    }
    if let Some(c) = &options.contrast {
        if c.len() != design.cols {
            return Err(Error(
                "the contrast needs one value per design column".into(),
            ));
        }
        o.contrasts = c.as_ptr();
        o.contrast_rows = 1;
    }
    if let Some(a) = options.alpha {
        o.alpha = a;
    }
    if let Some(t) = options.threads {
        o.threads = t;
    }
    o.legacy_cooks = options.legacy_cooks as c_int;
    let mut raw = std::ptr::null_mut();
    check(unsafe { ffi::rapidglm_bulk_de(&design.view(), &sparse, &o, &mut raw) })?;
    Ok(Table { raw })
}

/// NB fits at a fixed dispersion per gene with log size-factor offsets
/// (`rapidglm_nb_fixed_fits`).
pub struct FixedFits {
    /// genes x design columns, natural log scale.
    pub coefficients: Matrix,
    /// samples x genes: s_i exp(x_i' beta).
    pub fitted: Matrix,
    /// samples x genes: hat values of W^1/2 X (X'WX)^-1 X'W^1/2.
    pub hat: Matrix,
    pub converged: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub fn nb_fixed_fits(
    design: &Matrix,
    counts: &Matrix,
    size_factors: &[f64],
    dispersions: &[f64],
    tolerance: f64,
    max_iterations: usize,
    min_beta: f64,
    max_beta: f64,
    threads: i32,
) -> Result<FixedFits> {
    let (n, genes, p) = (counts.rows, counts.cols, design.cols);
    if design.rows != n || size_factors.len() != n || dispersions.len() != genes {
        return Err(Error("nb_fixed_fits: inconsistent dimensions".into()));
    }
    let mut coefficients = Matrix::filled(genes, p, f64::NAN);
    let mut fitted = Matrix::filled(n, genes, f64::NAN);
    let mut hat = Matrix::filled(n, genes, f64::NAN);
    let mut converged = vec![0 as c_int; genes];
    check(unsafe {
        ffi::rapidglm_nb_fixed_fits(
            &design.view(),
            &counts.view(),
            size_factors.as_ptr(),
            dispersions.as_ptr(),
            tolerance,
            max_iterations,
            min_beta,
            max_beta,
            threads,
            coefficients.values.as_mut_ptr(),
            fitted.values.as_mut_ptr(),
            hat.values.as_mut_ptr(),
            converged.as_mut_ptr(),
        )
    })?;
    Ok(FixedFits {
        coefficients,
        fitted,
        hat,
        converged: converged.iter().map(|&c| c != 0).collect(),
    })
}

/// Dispersions at fixed means from `start` (`rapidglm_nb_dispersions`):
/// Cox-Reid adjusted when `cox_reid`, with a normal prior on log dispersion
/// centred at log(start) when `prior_variance` > 0. Returns (dispersions,
/// converged).
#[allow(clippy::too_many_arguments)]
pub fn nb_dispersions(
    design: &Matrix,
    counts: &Matrix,
    mu: &Matrix,
    start: &[f64],
    min_dispersion: f64,
    max_dispersion: f64,
    prior_variance: f64,
    cox_reid: bool,
    keep_minimum_start: bool,
    threads: i32,
) -> Result<(Vec<f64>, Vec<bool>)> {
    let genes = counts.cols;
    if design.rows != counts.rows
        || mu.rows != counts.rows
        || mu.cols != genes
        || start.len() != genes
    {
        return Err(Error("nb_dispersions: inconsistent dimensions".into()));
    }
    let mut out = vec![f64::NAN; genes];
    let mut converged = vec![0 as c_int; genes];
    check(unsafe {
        ffi::rapidglm_nb_dispersions(
            &design.view(),
            &counts.view(),
            &mu.view(),
            start.as_ptr(),
            min_dispersion,
            max_dispersion,
            prior_variance,
            cox_reid as c_int,
            keep_minimum_start as c_int,
            threads,
            out.as_mut_ptr(),
            converged.as_mut_ptr(),
        )
    })?;
    Ok((out, converged.iter().map(|&c| c != 0).collect()))
}
