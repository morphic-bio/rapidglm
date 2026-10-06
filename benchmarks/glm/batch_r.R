# Loops R comparators over the batch responses. Usage: batch_r.R dir tool family libs
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(MASS)})
dir <- args[1]; tool <- args[2]; family <- args[3]
meta <- fromJSON(file.path(dir, "meta.json"))
n <- meta$n; p <- meta$p; m <- meta$m
X <- matrix(readBin(file.path(dir, "X.f64"), "double", n * p), n, p, byrow = TRUE)
Y <- matrix(readBin(file.path(dir, "Y.f64"), "double", n * m), n, m)
if (startsWith(tool, "rapidglm")) invisible(loadNamespace("rapidglm"))
start <- proc.time()[["elapsed"]]
coefs <- matrix(NA_real_, m, p + 1)
for (j in seq_len(m)) {
  y <- Y[, j]
  f <- switch(paste(tool, family),
    "glm.fit poisson" = glm.fit(X, y, family = poisson()),
    "rapidglm poisson" = rapidglm::glm_fit(X, y, family = poisson()),
    "glm.nb nb" = suppressWarnings(glm.nb(y ~ X - 1)),
    "rapidglm nb" = suppressWarnings(rapidglm::glm_nb(y ~ X - 1)))
  coefs[j, ] <- c(as.numeric(f$coefficients), if (family == "nb") f$theta else NA)
}
seconds <- proc.time()[["elapsed"]] - start
write.table(coefs, file.path(dir, paste0("coef-", tool, "-", family, ".tsv")), sep = "\t",
            row.names = FALSE, col.names = FALSE)
cat(toJSON(list(seconds = seconds), auto_unbox = TRUE), "\n")
