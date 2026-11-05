# rapidglm's special functions: the C++ results of RapidGLM's functions bit for
# bit (tests/fixtures/vmath_known_values.tsv, written by rapidglm_special_functions
# values; lgamma, the C library's, is not in it), base R within the documented bounds plus an allowance for base R's
# own error, and base R's handling of NA, NaN and attributes.
library(rapidglm)
f <- list(log = special_log, log1p = special_log1p, exp = special_exp,
          lgamma = special_lgamma, digamma = special_digamma, trigamma = special_trigamma)
base <- list(log = log, log1p = log1p, exp = exp, lgamma = lgamma, digamma = digamma,
             trigamma = trigamma)
bound <- c(log = 1, log1p = 1, exp = 1, lgamma = 4, digamma = 3.7, trigamma = 2.8)
# Allowance for base R's own error (rapidglm's errors are checked against mpmath
# in tests/test_vmath.cpp). Base R's trigamma is off by about 5 ulp at x ~ 70
# and up to about 60 ulp below 1e-10 (6.6e-15 relative at x = 1.03e-14, against
# mpmath; rapidglm's 6e-17), hence its larger allowance.
allowance <- c(log = 2, log1p = 2, exp = 2, lgamma = 2, digamma = 2, trigamma = 64)
cat("special path:", special_path(), "\n")
# From the repository root, from bindings/r/tests, or RAPIDGLM_VMATH_VALUES.
candidates <- c(Sys.getenv("RAPIDGLM_VMATH_VALUES"), file.path("tests", "fixtures", "vmath_known_values.tsv"),
                file.path("..", "..", "..", "tests", "fixtures", "vmath_known_values.tsv"))
fixture <- c(Filter(function(f) nzchar(f) && file.exists(f), candidates), "")[1]
from_hex <- function(h) sapply(h, function(s) readBin(as.raw(strtoi(substring(s, seq(1, 15, 2), seq(2, 16, 2)), 16L)),
                                                       "double", size = 8, endian = "big"))
if (file.exists(fixture)) {
  known <- read.delim(fixture, colClasses = "character")
  stopifnot(setequal(unique(known$`function`), setdiff(names(f), "lgamma")))
  for (name in setdiff(names(f), "lgamma")) {
    k <- known[known$`function` == name, ]
    x <- from_hex(k$x)
    expected <- from_hex(k$value)
    got <- suppressWarnings(f[[name]](unname(x), division = "slow"))
    stopifnot(identical(unname(got), unname(expected)))
    cat(name, nrow(k), "values identical to C++\n")
  }
} else cat("tests/fixtures/vmath_known_values.tsv not found (bit-for-bit check skipped)\n")
ulp <- function(v) 2^(pmax(floor(log2(abs(v))), -1022) - 52)
set.seed(1)
# Up to 1e15: beyond, base R's trigamma loses accuracy (5e-14 relative at 1e285,
# where rapidglm's equals 1/x to the last place, as mpmath).
x <- c(10^runif(2000, -300, -5), runif(2000, 1e-5, 3), runif(2000, 3, 100), 10^runif(2000, 2, 15),
       rgeom(2000, 1 / 21) + 1 / 10^runif(2000, -8, 3))
for (name in names(f)) {
  arg <- switch(name, exp = runif(10000, -700, 700), log1p = c(-runif(1000), x), x)
  ours <- f[[name]](arg)
  ref <- suppressWarnings(base[[name]](arg)) # base R's trigamma gives NaN below about 1e-150
  ok <- is.finite(ref) & ref != 0
  # Base R's lgamma and digamma keep absolute, not relative, accuracy next to
  # their zeros (lgamma at x = 1.00033: 2.5e-13 relative error against mpmath,
  # rapidglm's 1e-17; digamma near its root likewise): they are compared in
  # units of the last place of max(|f|, 1).
  scale <- if (name %in% c("lgamma", "digamma")) pmax(abs(ref[ok]), 1) else ref[ok]
  difference <- max(abs(ours[ok] - ref[ok]) / ulp(scale))
  cat(sprintf("%-8s max |rapidglm - base R| = %.3f ulp\n", name, difference))
  stopifnot(difference <= bound[[name]] + allowance[[name]])
}
m <- matrix(c(0.5, NA, NaN, 4), 2, dimnames = list(c("a", "b"), NULL))
for (name in names(f)) {
  out <- f[[name]](m)
  stopifnot(identical(dim(out), dim(m)), identical(dimnames(out), dimnames(m)),
            is.na(out[2]) && !is.nan(out[2]), is.nan(out[3]))
}
stopifnot(identical(special_lgamma(1:3), lgamma(1:3)) || max(abs(special_lgamma(1:3) - lgamma(1:3))) < 1e-15)
w <- tryCatch(special_digamma(-1), warning = function(w) conditionMessage(w))
stopifnot(identical(w, "NaNs produced"))
cat("special functions ok\n")

# Scoped mode selection survives a native error and leaves the previous mode intact.
for (mode in c("slow", "fast")) {
  z <- seq(0.1, 5, length.out = 1003)
  expected <- special_digamma(z, division = mode)
  stopifnot(all(is.finite(expected)))
  failed <- try(special_log(list(1), division = mode), silent = TRUE)
  stopifnot(inherits(failed, "try-error"))
  fit <- rapidglm::glm(y ~ x, data = data.frame(y = 1:8, x = rep(0:1, 4)),
                      family = poisson(), division = mode)
  stopifnot(fit$converged)
}
stopifnot(inherits(try(special_log(1, division = "unknown"), silent = TRUE), "try-error"))
