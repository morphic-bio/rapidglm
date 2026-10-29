# rapidglm (R)

A fitting method for `stats::glm()` and a `MASS::glm.nb()`-compatible negative
binomial fit, plus bulk differential expression and SCEPTRE-method
perturbation analysis, backed by the RapidGLM engine.

## Applications

```r
res <- rapidglm::bulk_de(counts, ~ cell + dex, col_data, contrast = c("dex", "trt", "untrt"),
                         lfc_shrink = "normal", threads = 8)   # counts: genes x samples
attr(res, "run")$size_factors
pairs_res <- rapidglm::sceptre(responses, grna_calls, ~ log_umi + batch, guides, pairs,
                               cell_data = covariates, moi = "high", threads = 8)
# every target against every response, without a list of pairs
grid_res <- rapidglm::sceptre_grid(responses, grna_calls, ~ log_umi + batch, guides,
                                   discovery_targets, negative_targets,
                                   cell_data = covariates, moi = "high", threads = 8)
attr(grid_res, "tally")   # the pairs that failed pair QC, by target and kind
```

- `bulk_de()` runs the DESeq2 procedure (written from its paper and
  documentation) and `edger_ql()` edgeR 4's default quasi-likelihood workflow
  (written from its papers, documentation and black-box runs). `sceptre()` runs the perturbation workflow on a list of
  pairs, and `sceptre_grid()` on targets and responses without a list. A pair's
  draws depend only on the seed, its response and its target (`response_ids`
  names the responses), so its row does not depend on what else is tested.
- Both call the RapidGLM C interface, which the package compiles in, so they
  run the same code as the CLI and the Python package.
- Inputs follow R conventions:
  - `counts` and `responses` are genes x observations (dense or `Matrix`);
  - the design is a matrix, or a formula with `col_data`/`cell_data`;
  - coefficients are given by name or 1-based index;
  - contrasts are `c(variable, tested, reference)` or a weight vector.
- Results are data frames whose columns match the CLI tables. Run-level values
  are in `attr(, "run")`.
- `tests/applications.R` checks both against the CLI:
  - `bulk_de()` bit for bit;
  - `sceptre()` to 1e-10, because R's BLAS can round differently from the
    CLI's OpenBLAS. Approximation-tail p-values below 1e-12 only need to be
    small on both sides.

The native application and GLM entry points accept `division = "slow"` or
`"fast"`; `NULL` inherits `RAPIDGLM_DIVISION`, read on first native use.
The build default is `slow`. Explicit choices are scoped to the call and
restored on return or error.

## Special functions

`special_log()`, `special_log1p()`, `special_exp()`, `special_digamma()` and
`special_trigamma()` (RapidGLM's; digamma and trigamma for x >= 0) and
`special_lgamma()` (libm in slow mode; the own positive kernel in fast mode) evaluate the engine's special functions on
double vectors, with the same `division` argument, keeping attributes and treating NA and NaN as base R does
([man/special.Rd](man/special.Rd)). `tests/special.R` checks RapidGLM's against
the C++ values bit for bit and all of them against base R.

## glm drop-ins

```r
fit <- glm(y ~ x + offset(log(depth)), family = poisson, data = d,
           method = rapidglm::glm_fit)          # any existing glm() call
fit <- rapidglm::glm(y ~ x, family = binomial, data = d)   # same, as a wrapper
nb  <- rapidglm::glm_nb(counts ~ group, data = d)          # class "negbin"
```

The result is an ordinary `glm` (or `negbin`) object, so `summary()`, `anova()`,
`predict(se.fit = TRUE)`, `vcov()`, `confint()`, `drop1()`, residuals and
influence measures work unchanged. Poisson (log), binomial (logit) and MASS negative binomial (log) families are
fitted natively. Gaussian identity fits, which are a single least-squares solve
that `glm.fit` performs with the QR `glm()` stores anyway, other families and
links, and supplied `etastart`/`mustart` use `stats::glm.fit()`.
`library(rapidglm)` masks `stats::glm` with the wrapper.

Differences from `glm.fit`: convergence also requires the coefficients to
settle (tolerance `min(1e-9, epsilon)`), and standard errors use the working
weights at the final estimates rather than the weights of the last iteration's
start. Aliasing uses R's threshold, `min(1e-7, epsilon / 1000)`.

The package uses R's own LAPACK. Inside the RapidGLM repository, install with
`R CMD INSTALL --preclean bindings/r` (the configure script copies the engine
headers; `--preclean` removes objects of an earlier build, which would
otherwise be linked in unchanged after the headers change).
Boost headers are required. The configure script also copies `src/c_api.cpp`
and the package builds with OpenMP. Tests: `tests/parity.R` and
`tests/applications.R` (`RAPIDGLM_CLI=build/rapidglm`).
