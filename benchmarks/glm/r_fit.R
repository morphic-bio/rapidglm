# Black-box R comparators for the GLM comparison suite (no package source is read).
# Usage: Rscript r_fit.R problem_dir tool output.json max_seconds [library]
args <- commandArgs(TRUE)
if (length(args) >= 5) .libPaths(c(normalizePath(strsplit(args[5], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(MASS)})
dir <- args[1]; tool <- args[2]; output <- args[3]; budget <- as.numeric(args[4])
meta <- fromJSON(file.path(dir, "meta.json"))
n <- meta$n; p <- meta$p
X <- matrix(readBin(file.path(dir, "X.f64"), "double", n * p, endian = "little"), n, p, byrow = TRUE)
y <- readBin(file.path(dir, "y.f64"), "double", n, endian = "little")
off <- readBin(file.path(dir, "offset.f64"), "double", n, endian = "little")
w <- readBin(file.path(dir, "weights.f64"), "double", n, endian = "little")
colnames(X) <- paste0("x", seq_len(p) - 1)
fam <- switch(meta$family, gaussian = gaussian(), poisson = poisson(), binomial = binomial(), nb = NULL)
# Load comparator namespaces before timing so loading cost is excluded.
if (tool %in% c("fastglm", "speedglm")) invisible(loadNamespace(tool))
if (startsWith(tool, "r-rapidglm")) invisible(loadNamespace("rapidglm"))
fit_once <- function() {
  switch(tool,
    "glm.fit" = glm.fit(X, y, weights = w, offset = off, family = fam),
    "glm" = glm(y ~ X - 1 + offset(off), family = fam, weights = w),
    "glm.nb" = glm.nb(y ~ X - 1 + offset(off), weights = w),
    "fastglm" = fastglm::fastglm(X, y, weights = w, offset = off, family = fam),
    "speedglm" = speedglm::speedglm.wfit(y, X, weights = w, offset = off, family = fam, intercept = FALSE),
    "r-rapidglm.fit" = rapidglm::glm_fit(X, y, weights = w, offset = off, family = fam),
    "r-rapidglm" = stats::glm(y ~ X - 1 + offset(off), family = fam, weights = w, method = rapidglm::glm_fit),
    "r-rapidglm.nb" = rapidglm::glm_nb(y ~ X - 1 + offset(off), weights = w),
    stop("unknown tool"))
}
seconds <- c(); warnings_seen <- character(); err <- NULL; fit <- NULL
repeat {
  start <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(tryCatch(fit_once(), error = function(e) { err <<- conditionMessage(e); NULL }),
    warning = function(wn) { warnings_seen <<- unique(c(warnings_seen, conditionMessage(wn))); invokeRestart("muffleWarning") })
  seconds <- c(seconds, proc.time()[["elapsed"]] - start)
  if (is.null(fit) || length(seconds) >= 7 || sum(seconds) >= budget) break
}
out <- list(tool = tool, seconds = seconds, warnings = warnings_seen, error = err)
if (!is.null(fit)) {
  coef_se <- function(object) {
    if (tool %in% c("glm.fit", "r-rapidglm.fit")) {
      class(object) <- c("glm", "lm")
      object$call <- quote(glm.fit())
    }
    tab <- tryCatch(coef(summary(object)), error = function(e) NULL)
    b <- as.numeric(coef(object))
    se <- rep(NA_real_, p)
    if (!is.null(tab)) {
      keep <- !is.na(b)
      se[keep] <- tab[, 2]
    }
    list(b = b, se = se)
  }
  cs <- if (tool == "fastglm") list(b = as.numeric(fit$coefficients), se = as.numeric(fit$se)) else
        if (tool == "speedglm") { s <- summary(fit); list(b = as.numeric(fit$coefficients), se = as.numeric(s$coefficients[, 2])) } else
        coef_se(fit)
  ll <- if (tool %in% c("glm.nb", "r-rapidglm.nb")) as.numeric(logLik(fit)) else if (tool %in% c("glm", "glm.fit", "r-rapidglm", "r-rapidglm.fit")) {
    obj <- fit; if (tool %in% c("glm.fit", "r-rapidglm.fit")) class(obj) <- c("glm", "lm")
    if (meta$family %in% c("poisson", "binomial")) -obj$aic / 2 + obj$rank else as.numeric(logLik(if (tool %in% c("glm", "r-rapidglm")) fit else { obj$call <- quote(glm.fit()); obj }))
  } else NA_real_
  out$coefficients <- cs$b; out$standard_errors <- cs$se; out$loglik <- ll
  out$converged <- if (!is.null(fit$converged)) isTRUE(fit$converged) else NA
  out$iterations <- if (!is.null(fit$iter)) fit$iter else NA
  out$theta <- if (tool %in% c("glm.nb", "r-rapidglm.nb")) fit$theta else NA
  out$boundary <- if (!is.null(fit$boundary)) isTRUE(fit$boundary) else NA
  out$rank <- if (!is.null(fit$rank)) fit$rank else NA
}
out$versions <- list(R = R.version.string, MASS = as.character(packageVersion("MASS")),
  package = if (tool %in% c("fastglm", "speedglm")) as.character(packageVersion(tool)) else NULL)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), output)
