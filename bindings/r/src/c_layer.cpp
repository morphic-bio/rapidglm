// R entry points for bulk_de, edger_ql, sceptre, sceptre_grid and qtl through the RapidGLM C
// interface (rapidglm_c.h; src/c_api.cpp is compiled into the package by configure). The R layer
// (R/applications.R) orients inputs and names options; this file fills the C option structs and
// converts result tables to R lists.
#include <cpp11.hpp>
#include <rapidglm/rapidglm_c.h>

#include <cmath>
#include <string>
#include <vector>

namespace {
struct Dense {
    std::vector<double> values; // row-major
    rapidglm_dense view;
};
Dense dense(const cpp11::doubles_matrix<> &m) {
    Dense d;
    const std::size_t n = m.nrow(), p = m.ncol();
    d.values.resize(n * p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            d.values[i * p + j] = m(i, j);
    d.view = {n, p, d.values.data()};
    return d;
}
// A CSC matrix (observations x features) from the slots of a dgCMatrix.
struct Sparse {
    std::vector<size_t> ptr, index;
    std::vector<double> values;
    rapidglm_sparse view;
};
Sparse sparse(const cpp11::integers &p, const cpp11::integers &i, const cpp11::doubles &x, int rows,
              int cols) {
    Sparse s;
    s.ptr.assign(p.begin(), p.end());
    s.index.assign(i.begin(), i.end());
    s.values.assign(x.begin(), x.end());
    s.view = {static_cast<size_t>(rows), static_cast<size_t>(cols), s.ptr.data(), s.index.data(),
              s.values.data()};
    return s;
}
bool has(const cpp11::list &o, const char *name) {
    const cpp11::strings names(o.names());
    for (const auto &n : names)
        if (std::string(n) == name)
            return true;
    return false;
}
double number(const cpp11::list &o, const char *name) {
    return cpp11::as_cpp<double>(o[name]);
}
cpp11::writable::list table(rapidglm_table *t) {
    using namespace cpp11::literals;
    const size_t rows = rapidglm_table_rows(t);
    cpp11::writable::list columns;
    cpp11::writable::strings column_names;
    for (size_t k = 0; k < rapidglm_table_columns(t); ++k) {
        const char *name = rapidglm_table_column_name(t, k);
        if (rapidglm_table_column_type(t, k) == RAPIDGLM_COLUMN_NUMERIC) {
            const double *v = rapidglm_table_numeric(t, name);
            columns.push_back(cpp11::writable::doubles(v, v + rows));
        } else {
            cpp11::writable::strings s(static_cast<R_xlen_t>(rows));
            for (size_t r = 0; r < rows; ++r)
                s[static_cast<R_xlen_t>(r)] = rapidglm_table_text(t, name, r);
            columns.push_back(s);
        }
        column_names.push_back(name);
    }
    columns.names() = column_names;
    cpp11::writable::list scalars, vectors;
    cpp11::writable::strings scalar_names, vector_names;
    for (size_t k = 0; k < rapidglm_table_scalars(t); ++k) {
        const char *name = rapidglm_table_scalar_name(t, k);
        scalars.push_back(cpp11::as_sexp(rapidglm_table_scalar(t, name)));
        scalar_names.push_back(name);
    }
    for (size_t k = 0; k < rapidglm_table_vectors(t); ++k) {
        const char *name = rapidglm_table_vector_name(t, k);
        size_t n = 0;
        const double *v = rapidglm_table_vector(t, name, &n);
        vectors.push_back(cpp11::writable::doubles(v, v + n));
        vector_names.push_back(name);
    }
    // cpp11 cannot name an empty list.
    if (scalars.size())
        scalars.names() = scalar_names;
    if (vectors.size())
        vectors.names() = vector_names;
    rapidglm_table_free(t);
    return cpp11::writable::list(
        {"columns"_nm = columns, "scalars"_nm = scalars, "vectors"_nm = vectors});
}
void check(int status) {
    if (status != RAPIDGLM_OK)
        cpp11::stop("%s", rapidglm_last_error());
}
// R strings as the C interface takes them; the pointers refer to `values`.
struct Texts {
    std::vector<std::string> values;
    std::vector<const char *> pointers;
    explicit Texts(const cpp11::strings &s) : values(s.begin(), s.end()) {
        for (const auto &v : values)
            pointers.push_back(v.c_str());
    }
    Texts(const Texts &) = delete;
    Texts &operator=(const Texts &) = delete;
};
// The targets and responses of a grid request (sceptre_grid).
struct Grid {
    cpp11::strings discovery_targets, negative_targets;
    cpp11::integers responses; // empty: every response
};
// sceptre and sceptre_grid: with a grid the result is list(result, tally).
cpp11::writable::list
run_sceptre(const cpp11::doubles_matrix<> &design, const cpp11::integers &yp,
            const cpp11::integers &yi, const cpp11::doubles &yx, int genes,
            const cpp11::integers &gp, const cpp11::integers &gi, const cpp11::doubles &gx,
            int n_guides, const cpp11::strings &guide_ids, const cpp11::strings &guide_targets,
            const cpp11::integers &guide_non_targeting, const cpp11::integers &pair_responses,
            const cpp11::strings &pair_targets, const cpp11::integers &pair_kinds,
            const cpp11::list &o, const Grid *grid) {
    using namespace cpp11::literals;
    const auto d = dense(design);
    const int cells = static_cast<int>(d.view.rows);
    const auto y = sparse(yp, yi, yx, cells, genes);
    const auto g = sparse(gp, gi, gx, cells, n_guides);
    rapidglm_sceptre_options s;
    rapidglm_sceptre_options_init(&s, has(o, "high_moi") && number(o, "high_moi") != 0);
    const auto integer = [&](const char *name, int &field) {
        if (has(o, name))
            field = static_cast<int>(number(o, name));
    };
    const auto real = [&](const char *name, double &field) {
        if (has(o, name))
            field = number(o, name);
    };
    integer("family", s.model.family);
    integer("null_model", s.null_model);
    integer("nt_controls", s.nt_controls);
    integer("crt", s.crt);
    integer("integration", s.integration);
    integer("side", s.side);
    integer("approximate", s.approximate);
    integer("effect_fits", s.effect_fits);
    integer("cell_qc", s.cell_qc);
    integer("assignment_qc", s.assignment_qc);
    integer("bonferroni", s.bonferroni);
    integer("crt_draw", s.crt_draw);
    integer("fit_check", s.fit_check);
    integer("threads", s.threads);
    integer("stream_keys", s.stream_keys);
    integer("adjustment_family", s.family);
    integer("null_start", s.null_start);
    integer("batch_scoring", s.batch_scoring);
    real("screen", s.screen);
    real("alpha", s.alpha);
    real("lower_quantile", s.lower_quantile);
    real("upper_quantile", s.upper_quantile);
    real("max_mito", s.max_mito);
    if (has(o, "min_nonzero"))
        s.min_nonzero = static_cast<size_t>(number(o, "min_nonzero"));
    if (has(o, "confirm_draws"))
        s.confirm_draws = static_cast<size_t>(number(o, "confirm_draws"));
    if (has(o, "seed"))
        s.seed = static_cast<uint64_t>(number(o, "seed"));
    std::vector<size_t> budgets;
    if (has(o, "budgets")) {
        cpp11::doubles v(o["budgets"]);
        for (double b : v)
            budgets.push_back(static_cast<size_t>(b));
        s.budgets = budgets.data();
        s.n_budgets = budgets.size();
    }
    if (has(o, "family_size")) {
        cpp11::doubles v(o["family_size"]); // discovery, negative, positive
        if (v.size() != 3)
            cpp11::stop("family_size needs three sizes");
        for (int k = 0; k < 3; ++k)
            s.family_size[k] = static_cast<size_t>(v[k]);
    }
    const Texts response_ids(has(o, "response_ids") ? cpp11::strings(o["response_ids"])
                                                    : cpp11::strings());
    if (has(o, "response_ids")) {
        // The C interface reads one ID per response column.
        if (response_ids.values.size() != static_cast<size_t>(genes))
            cpp11::stop("response_ids needs one ID per response");
        s.response_ids = response_ids.pointers.data();
    }
    const Texts ids(guide_ids), targets(guide_targets), pair_target(pair_targets);
    std::vector<int> nt(guide_non_targeting.begin(), guide_non_targeting.end()),
        kinds(pair_kinds.begin(), pair_kinds.end());
    std::vector<size_t> responses(pair_responses.begin(), pair_responses.end());
    rapidglm_table *t = nullptr;
    if (!grid) {
        check(rapidglm_sceptre(&d.view, &y.view, &g.view, static_cast<size_t>(n_guides),
                               ids.pointers.data(), targets.pointers.data(), nt.data(),
                               responses.size(), responses.data(), pair_target.pointers.data(),
                               kinds.data(), &s, &t));
        return table(t);
    }
    const Texts discovery(grid->discovery_targets), negative(grid->negative_targets);
    std::vector<size_t> grid_responses(grid->responses.begin(), grid->responses.end());
    rapidglm_table *tally = nullptr;
    check(rapidglm_sceptre_grid(
        &d.view, &y.view, &g.view, static_cast<size_t>(n_guides), ids.pointers.data(),
        targets.pointers.data(), nt.data(), discovery.values.size(), discovery.pointers.data(),
        negative.values.size(), negative.pointers.data(), grid_responses.size(),
        grid_responses.data(), responses.size(), responses.data(), pair_target.pointers.data(),
        kinds.data(), &s, &t, &tally));
    return cpp11::writable::list({"result"_nm = table(t), "tally"_nm = table(tally)});
}
} // namespace

[[cpp11::register]] cpp11::writable::list rapidglm_bulk_de_(cpp11::doubles_matrix<> design,
                                                            cpp11::integers p, cpp11::integers i,
                                                            cpp11::doubles x, int samples,
                                                            int genes, cpp11::list o) {
    const auto d = dense(design);
    const auto y = sparse(p, i, x, samples, genes);
    rapidglm_bulk_options b;
    rapidglm_bulk_options_init(&b);
    std::vector<double> contrasts, size_factors;
    if (has(o, "coefficient")) {
        const double index = number(o, "coefficient");
        if (!std::isfinite(index) || index < 0 || index >= design.ncol() ||
            std::floor(index) != index)
            cpp11::stop("coefficient outside the design");
        b.coefficient = static_cast<size_t>(index);
    }
    if (has(o, "contrasts")) {
        cpp11::doubles_matrix<> c(o["contrasts"]); // rows x design columns
        if (c.ncol() != design.ncol())
            cpp11::stop("contrast length differs from design");
        for (int r = 0; r < c.nrow(); ++r)
            for (int k = 0; k < c.ncol(); ++k)
                contrasts.push_back(c(r, k));
        b.contrasts = contrasts.data();
        b.contrast_rows = static_cast<size_t>(c.nrow());
    }
    if (has(o, "size_factor_values")) {
        cpp11::doubles s(o["size_factor_values"]);
        if (s.size() != samples)
            cpp11::stop("size_factor_values needs one value per sample");
        size_factors.assign(s.begin(), s.end());
        b.size_factors = size_factors.data();
    }
    const auto integer = [&](const char *name, int &field) {
        if (has(o, name))
            field = static_cast<int>(number(o, name));
    };
    const auto real = [&](const char *name, double &field) {
        if (has(o, name))
            field = number(o, name);
    };
    integer("lrt", b.lrt);
    integer("size_factor_method", b.size_factor_method);
    integer("trend", b.trend);
    integer("keep_minimum_start", b.keep_minimum_start);
    integer("cooks_filter", b.cooks_filter);
    integer("legacy_cooks", b.legacy_cooks);
    integer("independent_filtering", b.independent_filtering);
    integer("lfc_shrinkage", b.lfc_shrinkage);
    integer("threads", b.threads);
    real("min_mean", b.min_mean);
    real("outlier_sd", b.outlier_sd);
    real("cooks_cutoff", b.cooks_cutoff);
    real("alpha", b.alpha);
    real("min_dispersion", b.min_dispersion);
    real("max_dispersion", b.max_dispersion);
    if (has(o, "min_replicates_for_replace"))
        b.min_replicates_for_replace = static_cast<size_t>(number(o, "min_replicates_for_replace"));
    if (has(o, "seed"))
        b.seed = static_cast<uint64_t>(number(o, "seed"));
    rapidglm_table *t = nullptr;
    check(rapidglm_bulk_de(&d.view, &y.view, &b, &t));
    return table(t);
}

[[cpp11::register]] cpp11::writable::list rapidglm_edger_ql_(cpp11::doubles_matrix<> design,
                                                             cpp11::integers p, cpp11::integers i,
                                                             cpp11::doubles x, int samples,
                                                             int genes, cpp11::list o) {
    const auto d = dense(design);
    const auto y = sparse(p, i, x, samples, genes);
    rapidglm_ql_options q;
    rapidglm_ql_options_init(&q);
    std::vector<double> contrasts, factors;
    if (has(o, "coefficient")) {
        const double index = number(o, "coefficient");
        if (!std::isfinite(index) || index < 0 || index >= design.ncol() ||
            std::floor(index) != index)
            cpp11::stop("coefficient outside the design");
        q.coefficient = static_cast<size_t>(index);
    }
    if (has(o, "contrasts")) {
        cpp11::doubles_matrix<> c(o["contrasts"]); // rows x design columns
        if (c.ncol() != design.ncol())
            cpp11::stop("contrast length differs from design");
        for (int r = 0; r < c.nrow(); ++r)
            for (int k = 0; k < c.ncol(); ++k)
                contrasts.push_back(c(r, k));
        q.contrasts = contrasts.data();
        q.contrast_rows = static_cast<size_t>(c.nrow());
    }
    if (has(o, "norm_factors")) {
        cpp11::doubles f(o["norm_factors"]);
        if (f.size() != samples)
            cpp11::stop("norm_factors needs one value per sample");
        factors.assign(f.begin(), f.end());
        q.norm_factors = factors.data();
    }
    const auto integer = [&](const char *name, int &field) {
        if (has(o, name))
            field = static_cast<int>(number(o, name));
    };
    const auto real = [&](const char *name, double &field) {
        if (has(o, name))
            field = number(o, name);
    };
    integer("filter", q.filter);
    integer("tmm", q.tmm);
    integer("threads", q.threads);
    real("min_count", q.min_count);
    real("min_total_count", q.min_total_count);
    real("large_n", q.large_n);
    real("min_prop", q.min_prop);
    real("dispersion", q.dispersion);
    real("prior_count", q.prior_count);
    rapidglm_table *t = nullptr;
    check(rapidglm_edger_ql(&d.view, &y.view, &q, &t));
    return table(t);
}

[[cpp11::register]] cpp11::writable::list
rapidglm_sceptre_(cpp11::doubles_matrix<> design, cpp11::integers yp, cpp11::integers yi,
                  cpp11::doubles yx, int genes, cpp11::integers gp, cpp11::integers gi,
                  cpp11::doubles gx, int n_guides, cpp11::strings guide_ids,
                  cpp11::strings guide_targets, cpp11::integers guide_non_targeting,
                  cpp11::integers pair_responses, cpp11::strings pair_targets,
                  cpp11::integers pair_kinds, cpp11::list o) {
    return run_sceptre(design, yp, yi, yx, genes, gp, gi, gx, n_guides, guide_ids, guide_targets,
                       guide_non_targeting, pair_responses, pair_targets, pair_kinds, o, nullptr);
}

// grid_responses: 0-based response columns (empty: every response). Returns
// list(result, tally), each as rapidglm_sceptre_ returns its table.
[[cpp11::register]] cpp11::writable::list
rapidglm_sceptre_grid_(cpp11::doubles_matrix<> design, cpp11::integers yp, cpp11::integers yi,
                       cpp11::doubles yx, int genes, cpp11::integers gp, cpp11::integers gi,
                       cpp11::doubles gx, int n_guides, cpp11::strings guide_ids,
                       cpp11::strings guide_targets, cpp11::integers guide_non_targeting,
                       cpp11::strings discovery_targets, cpp11::strings negative_targets,
                       cpp11::integers grid_responses, cpp11::integers pair_responses,
                       cpp11::strings pair_targets, cpp11::integers pair_kinds, cpp11::list o) {
    const Grid grid{discovery_targets, negative_targets, grid_responses};
    return run_sceptre(design, yp, yi, yx, genes, gp, gi, gx, n_guides, guide_ids, guide_targets,
                       guide_non_targeting, pair_responses, pair_targets, pair_kinds, o, &grid);
}

// Cis-QTL mapping. phenotypes: samples x phenotypes and genotypes: samples x
// variants, as R stores them (column-major), which is the C interface's layout
// of one phenotype's values and one variant's calls in a row. Genotypes are
// integer, logical or double calls (0, 1, 2; NA or negative: missing). Empty
// allele vectors, a covariate matrix without columns and empty library sizes
// mean "none". Returns list(result, nominal), each as rapidglm_sceptre_ returns
// its table.
[[cpp11::register]] cpp11::writable::list
rapidglm_qtl_(cpp11::doubles phenotypes, int samples, cpp11::strings phenotype_ids,
              cpp11::strings phenotype_chrom, cpp11::doubles phenotype_start,
              cpp11::doubles phenotype_end, SEXP genotypes, cpp11::strings variant_ids,
              cpp11::strings variant_chrom, cpp11::doubles variant_position, cpp11::strings allele1,
              cpp11::strings allele2, cpp11::doubles_matrix<> covariates,
              cpp11::doubles library_sizes, cpp11::list o) {
    using namespace cpp11::literals;
    const size_t n = static_cast<size_t>(samples), count = phenotype_ids.size(),
                 variants = variant_ids.size();
    if (n == 0 || phenotypes.size() != n * count)
        cpp11::stop("phenotypes need one column per phenotype and one row per sample");
    const auto integral = [](const cpp11::doubles &v, const char *name) {
        std::vector<int64_t> out(v.size());
        for (R_xlen_t i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v[i]) || std::floor(v[i]) != v[i] || std::fabs(v[i]) > 9e15)
                cpp11::stop("%s must be integers", name);
            out[static_cast<size_t>(i)] = static_cast<int64_t>(v[i]);
        }
        return out;
    };
    if (phenotype_chrom.size() != static_cast<R_xlen_t>(count) ||
        phenotype_start.size() != static_cast<R_xlen_t>(count) ||
        phenotype_end.size() != static_cast<R_xlen_t>(count))
        cpp11::stop("phenotype annotations need one entry per phenotype");
    if (variant_chrom.size() != static_cast<R_xlen_t>(variants) ||
        variant_position.size() != static_cast<R_xlen_t>(variants))
        cpp11::stop("variant annotations need one entry per variant");
    if ((allele1.size() != 0 && allele1.size() != static_cast<R_xlen_t>(variants)) ||
        (allele2.size() != 0 && allele2.size() != static_cast<R_xlen_t>(variants)))
        cpp11::stop("alleles need one entry per variant");
    const auto starts = integral(phenotype_start, "start"), ends = integral(phenotype_end, "end"),
               positions = integral(variant_position, "position");
    // Calls, one byte each: 0, 1, 2 or -1 for missing.
    std::vector<int8_t> calls(variants * n);
    {
        SEXP g = genotypes;
        const R_xlen_t total = Rf_xlength(g);
        if (static_cast<size_t>(total) != variants * n)
            cpp11::stop("genotypes need one column per variant and one row per sample");
        const auto convert = [&](bool missing, double v, size_t k) {
            if (missing || v < 0)
                calls[k] = -1;
            else if (v == 0 || v == 1 || v == 2)
                calls[k] = static_cast<int8_t>(v);
            else
                cpp11::stop("genotype calls must be 0, 1 or 2 copies of allele 1, or missing");
        };
        if (TYPEOF(g) == INTSXP || TYPEOF(g) == LGLSXP) {
            const int *x = TYPEOF(g) == INTSXP ? INTEGER(g) : LOGICAL(g);
            for (R_xlen_t k = 0; k < total; ++k)
                convert(x[k] == NA_INTEGER, x[k], static_cast<size_t>(k));
        } else if (TYPEOF(g) == REALSXP) {
            const double *x = REAL(g);
            for (R_xlen_t k = 0; k < total; ++k)
                convert(std::isnan(x[k]), x[k], static_cast<size_t>(k));
        } else {
            cpp11::stop("genotypes must be numeric");
        }
    }
    const Texts ids(phenotype_ids), chrom(phenotype_chrom), vid(variant_ids), vchrom(variant_chrom),
        a1(allele1), a2(allele2);
    rapidglm_qtl_phenotypes p;
    p.values = {count, n, REAL(phenotypes)};
    p.id = ids.pointers.data();
    p.chrom = chrom.pointers.data();
    p.start = starts.data();
    p.end = ends.data();
    rapidglm_qtl_genotypes g;
    g.variants = variants;
    g.samples = n;
    g.calls = calls.data();
    g.chrom = vchrom.pointers.data();
    g.position = positions.data();
    g.id = vid.pointers.data();
    g.allele1 = allele1.size() ? a1.pointers.data() : nullptr;
    g.allele2 = allele2.size() ? a2.pointers.data() : nullptr;
    const auto z = covariates.ncol() ? dense(covariates) : Dense();
    if (covariates.ncol() && static_cast<size_t>(covariates.nrow()) != n)
        cpp11::stop("covariates need one row per sample");
    rapidglm_qtl_options q;
    rapidglm_qtl_options_init(&q);
    const auto integer = [&](const char *name, int &field) {
        if (has(o, name))
            field = static_cast<int>(number(o, name));
    };
    const auto real = [&](const char *name, double &field) {
        if (has(o, name))
            field = number(o, name);
    };
    integer("family", q.model.family);
    integer("dispersion", q.model.dispersion);
    integer("robust", q.model.robust);
    integer("cox_reid", q.model.cox_reid);
    integer("nominal", q.nominal);
    integer("threads", q.threads);
    integer("permutation_precision", q.permutation_precision);
    integer("permutation_fits", q.permutation_fits);
    real("theta", q.model.theta);
    real("tolerance", q.model.tolerance);
    real("ridge", q.model.ridge);
    real("min_maf", q.min_maf);
    if (has(o, "max_iterations"))
        q.model.max_iterations = static_cast<size_t>(number(o, "max_iterations"));
    if (has(o, "window"))
        q.window = static_cast<int64_t>(number(o, "window"));
    if (has(o, "permutations"))
        q.permutations = static_cast<size_t>(number(o, "permutations"));
    if (has(o, "seed"))
        q.seed = static_cast<uint64_t>(number(o, "seed"));
    if (library_sizes.size() != 0 && static_cast<size_t>(library_sizes.size()) != n)
        cpp11::stop("library_sizes needs one size per sample");
    rapidglm_table *t = nullptr, *pairs = nullptr;
    check(rapidglm_qtl(&p, &g, covariates.ncol() ? &z.view : nullptr,
                       library_sizes.size() ? REAL(library_sizes) : nullptr, &q, &t, &pairs));
    return cpp11::writable::list({"result"_nm = table(t), "nominal"_nm = table(pairs)});
}
