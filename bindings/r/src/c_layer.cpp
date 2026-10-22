// R entry points for bulk_de, sceptre and sceptre_grid through the RapidGLM C interface
// (rapidglm_c.h; src/c_api.cpp is compiled into the package by configure).
// The R layer (R/applications.R) orients inputs and names options; this file
// fills the C option structs and converts result tables to R lists.
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
