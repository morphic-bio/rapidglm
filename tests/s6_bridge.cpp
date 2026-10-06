// Validation-only Rcpp bridge; production library and CLI do not link R/Rcpp.
#include <Rcpp.h>
#include <rapidglm/rapidglm.hpp>
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::export]]
Rcpp::DataFrame rapidglm_s6(Rcpp::NumericMatrix responses, Rcpp::NumericMatrix covariates,
                            Rcpp::NumericVector treatment, double seed,
                            std::string null_model = "sceptre", std::string crt_draw = "thinned") {
    using namespace rapidglm;
    Matrix design(covariates.nrow(), covariates.ncol());
    for (std::size_t i = 0; i < design.rows; ++i)
        for (std::size_t j = 0; j < design.cols; ++j)
            design(i, j) = covariates(i, j);
    SparseCSC y;
    y.rows = responses.nrow();
    y.cols = responses.ncol();
    y.column_ptr = {0};
    for (std::size_t j = 0; j < y.cols; ++j) {
        for (std::size_t i = 0; i < y.rows; ++i)
            if (responses(i, j) != 0) {
                y.row_index.push_back(i);
                y.values.push_back(responses(i, j));
            }
        y.column_ptr.push_back(y.values.size());
    }
    SparseCSC g;
    g.rows = y.rows;
    g.cols = 1;
    g.column_ptr = {0};
    for (std::size_t i = 0; i < g.rows; ++i)
        if (treatment[i]) {
            g.row_index.push_back(i);
            g.values.push_back(1);
        }
    g.column_ptr.push_back(g.values.size());
    std::vector<Pair> pairs;
    for (std::size_t j = 0; j < y.cols; ++j)
        pairs.push_back({j, "target", PairKind::discovery});
    SceptreOptions options;
    options.high_moi = true;
    options.nt_controls = false;
    options.crt = true;
    options.cell_qc = false;
    options.min_nonzero = 0;
    options.resampling.seed = static_cast<std::uint64_t>(seed);
    options.resampling.budgets = {499, 4999};
    options.model.family = Family::negative_binomial;
    // direct/direct reproduces the workflow defaults before 2026-09-29.
    require(null_model == "sceptre" || null_model == "direct", "unknown S6 null model");
    require(crt_draw == "thinned" || crt_draw == "direct", "unknown S6 CRT draw");
    options.null_model = null_model == "sceptre" ? FitMethod::sceptre_null : FitMethod::direct;
    options.crt_draw = crt_draw == "thinned" ? CrtDraw::thinned : CrtDraw::direct;
    auto result = sceptre(design, y, g, {{"guide", "target", false}}, pairs, {}, options);
    Rcpp::NumericVector p(y.cols);
    Rcpp::CharacterVector status(y.cols);
    for (std::size_t j = 0; j < y.cols; ++j) {
        p[j] = result.tests[j].p_value;
        status[j] = std::isfinite(p[j]) ? "ok" : "invalid";
    }
    return Rcpp::DataFrame::create(Rcpp::_["p_value"] = p, Rcpp::_["status"] = status);
}
