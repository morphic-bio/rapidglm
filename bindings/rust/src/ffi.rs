//! Declarations from `include/rapidglm/rapidglm_c.h` (C API version 1). Only
//! the functions used by this crate are declared; the header is the reference.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

pub const RAPIDGLM_OK: c_int = 0;

#[repr(C)]
pub struct rapidglm_dense {
    pub rows: usize,
    pub cols: usize,
    pub values: *const f64, // rows x cols, row-major
}

#[repr(C)]
pub struct rapidglm_sparse {
    pub rows: usize,
    pub cols: usize,
    pub column_ptr: *const usize,
    pub row_index: *const usize,
    pub values: *const f64,
}

#[repr(C)]
pub struct rapidglm_table {
    _private: [u8; 0],
}

#[repr(C)]
pub struct rapidglm_bulk_options {
    pub struct_size: usize,
    pub coefficient: usize,
    pub contrasts: *const f64,
    pub contrast_rows: usize,
    pub lrt: c_int,
    pub size_factor_method: c_int,
    pub size_factors: *const f64,
    pub trend: c_int,
    pub min_mean: f64,
    pub outlier_sd: f64,
    pub cooks_cutoff: f64,
    pub alpha: f64,
    pub keep_minimum_start: c_int,
    pub cooks_filter: c_int,
    pub independent_filtering: c_int,
    pub min_replicates_for_replace: usize,
    pub lfc_shrinkage: c_int,
    pub seed: u64,
    pub threads: c_int,
    pub min_dispersion: f64,
    pub max_dispersion: f64,
    pub legacy_cooks: c_int,
}

extern "C" {
    pub fn rapidglm_version() -> *const c_char;
    pub fn rapidglm_c_api_version() -> c_int;
    pub fn rapidglm_last_error() -> *const c_char;

    pub fn rapidglm_table_rows(table: *const rapidglm_table) -> usize;
    pub fn rapidglm_table_columns(table: *const rapidglm_table) -> usize;
    pub fn rapidglm_table_column_name(table: *const rapidglm_table, column: usize)
        -> *const c_char;
    pub fn rapidglm_table_column_type(table: *const rapidglm_table, column: usize) -> c_int;
    pub fn rapidglm_table_numeric(table: *const rapidglm_table, name: *const c_char) -> *const f64;
    pub fn rapidglm_table_text(
        table: *const rapidglm_table,
        name: *const c_char,
        row: usize,
    ) -> *const c_char;
    pub fn rapidglm_table_scalar(table: *const rapidglm_table, name: *const c_char) -> f64;
    pub fn rapidglm_table_vector(
        table: *const rapidglm_table,
        name: *const c_char,
        length: *mut usize,
    ) -> *const f64;
    pub fn rapidglm_table_free(table: *mut rapidglm_table);

    pub fn rapidglm_bulk_options_init(options: *mut rapidglm_bulk_options);
    pub fn rapidglm_bulk_de(
        design: *const rapidglm_dense,
        counts: *const rapidglm_sparse,
        options: *const rapidglm_bulk_options,
        result: *mut *mut rapidglm_table,
    ) -> c_int;

    pub fn rapidglm_nb_fixed_fits(
        design: *const rapidglm_dense,
        counts: *const rapidglm_dense,
        size_factors: *const f64,
        dispersions: *const f64,
        tolerance: f64,
        max_iterations: usize,
        min_beta: f64,
        max_beta: f64,
        threads: c_int,
        coefficients: *mut f64,
        fitted: *mut f64,
        hat: *mut f64,
        converged: *mut c_int,
    ) -> c_int;
    pub fn rapidglm_nb_dispersions(
        design: *const rapidglm_dense,
        counts: *const rapidglm_dense,
        mu: *const rapidglm_dense,
        start: *const f64,
        min_dispersion: f64,
        max_dispersion: f64,
        prior_variance: f64,
        cox_reid: c_int,
        keep_minimum_start: c_int,
        threads: c_int,
        dispersions: *mut f64,
        converged: *mut c_int,
    ) -> c_int;
}
