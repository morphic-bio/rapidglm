# Black-box edgeR run for the edgeR quasi-likelihood specification
# (validation/work/edger/EDGER_QL_SPEC.md). Uses only edgeR's and limma's public
# functions and records their documented outputs; no package source is read.
# Usage: reference_edger.R dataset_dir coefficient_name out_dir libs [threads] [unit_matrices]
# dataset_dir holds counts.mtx (genes x samples) and design.tsv (header = column names).
# Runs the default QL workflow of the edgeR 4 User's Guide (DGEList, filterByExpr on the
# design, recomputed library sizes, TMM normLibSizes, glmQLFit, glmQLFTest) and, as a second
# output, the classic LRT (estimateDisp, glmFit, glmLRT) on the same filtered DGEList.
# threads is recorded only: edgeR 4.6.2 has no thread control and R here uses the reference
# BLAS, so every step runs on one thread. unit_matrices = 1 adds an untimed
# glmQLFit(keep.unit.mat = TRUE) and writes its leverage and adjusted unit deviances and df.
# Timed: each step from DGEList() to topTags(); reading the files is not timed.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[4], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages({library(edgeR); library(Matrix); library(jsonlite)})
dir <- args[1]; coef <- args[2]; out <- args[3]
threads <- if (length(args) >= 5) as.integer(args[5]) else 1L
unit_matrices <- length(args) >= 6 && args[6] == "1"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
counts <- as.matrix(readMM(file.path(dir, "counts.mtx")))
storage.mode(counts) <- "double"
X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
if (!coef %in% colnames(X)) stop("coefficient ", coef, " is not a design column")
j <- match(coef, colnames(X))
colnames(counts) <- paste0("s", seq_len(ncol(counts)))
rownames(counts) <- paste0("g", seq_len(nrow(counts)) - 1)
rownames(X) <- colnames(counts)
seconds <- list()
tick <- function(name, expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- force(expr)
  seconds[[name]] <<- proc.time()[["elapsed"]] - t0
  value
}

# Quasi-likelihood workflow (User's Guide 1.4 and 2.12; filterByExpr given the design, 2.7).
y0 <- tick("dgelist", DGEList(counts))
keep <- tick("filter_by_expr", filterByExpr(y0, design = X))
y <- tick("subset", y0[keep, , keep.lib.sizes = FALSE])
y <- tick("norm_lib_sizes", normLibSizes(y))
fit <- tick("glm_ql_fit", glmQLFit(y, X))
ql <- tick("glm_ql_ftest", glmQLFTest(fit, coef = j))
ql_tab <- tick("top_tags_ql", topTags(ql, n = Inf, sort.by = "none")$table)
ql_steps <- c("dgelist", "filter_by_expr", "subset", "norm_lib_sizes", "glm_ql_fit", "glm_ql_ftest", "top_tags_ql")
ql_seconds <- sum(unlist(seconds[ql_steps]))

# Classic GLM likelihood-ratio workflow on the same filtered, normalized DGEList (2.11).
yd <- tick("estimate_disp", estimateDisp(y, X))
lfit <- tick("glm_fit", glmFit(yd, X))
lrt <- tick("glm_lrt", glmLRT(lfit, coef = j))
lrt_tab <- tick("top_tags_lrt", topTags(lrt, n = Inf, sort.by = "none")$table)
lrt_steps <- c("dgelist", "filter_by_expr", "subset", "norm_lib_sizes", "estimate_disp", "glm_fit", "glm_lrt", "top_tags_lrt")
lrt_seconds <- sum(unlist(seconds[lrt_steps]))

gene <- which(keep) - 1L
s2 <- fit$deviance.adj / fit$df.residual.adj
rep_len_n <- function(v) if (length(v) == 1L) rep(v, length(gene)) else v
# One-way designs are fitted without iterations, so iter and failed may be absent.
opt <- function(v) if (is.null(v)) rep(NA, length(gene)) else v
genes_ql <- data.frame(gene = gene, AveLogCPM = fit$AveLogCPM,
                       logFC = ql_tab$logFC, logCPM = ql_tab$logCPM, F = ql_tab$F,
                       PValue = ql_tab$PValue, FDR = ql_tab$FDR,
                       coefficient = fit$coefficients[, j],
                       unshrunk_coefficient = fit$unshrunk.coefficients[, j],
                       deviance = fit$deviance, df_residual = fit$df.residual,
                       deviance_adj = fit$deviance.adj, df_residual_adj = fit$df.residual.adj,
                       s2 = s2, s2_prior = rep_len_n(fit$s2.prior), s2_post = fit$s2.post,
                       df_prior = rep_len_n(fit$df.prior), df_total = ql$df.total,
                       iter = opt(fit$iter), failed = opt(fit$failed))
write.table(genes_ql, file.path(out, "genes_ql.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
genes_lrt <- data.frame(gene = gene, AveLogCPM = yd$AveLogCPM,
                        logFC = lrt_tab$logFC, logCPM = lrt_tab$logCPM, LR = lrt_tab$LR,
                        PValue = lrt_tab$PValue, FDR = lrt_tab$FDR,
                        coefficient = lfit$coefficients[, j],
                        unshrunk_coefficient = lfit$unshrunk.coefficients[, j],
                        trended_dispersion = yd$trended.dispersion,
                        tagwise_dispersion = yd$tagwise.dispersion,
                        deviance = lfit$deviance, iter = opt(lfit$iter), failed = opt(lfit$failed))
write.table(genes_lrt, file.path(out, "genes_lrt.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(gene = seq_len(nrow(counts)) - 1L, kept = as.integer(keep),
                       total = rowSums(counts)),
            file.path(out, "filter.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(sample = colnames(counts), lib_size_raw = colSums(counts),
                       lib_size = y$samples$lib.size, norm_factor = y$samples$norm.factors,
                       effective_lib_size = y$samples$lib.size * y$samples$norm.factors),
            file.path(out, "samples.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
# Mean-centred log coefficients of every design column (natural log), as glmQLFit reports.
write.table(data.frame(gene = gene, fit$coefficients, check.names = FALSE),
            file.path(out, "coefficients_ql.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

if (unit_matrices) {
  fit_unit <- glmQLFit(y, X, keep.unit.mat = TRUE)
  for (m in c("leverage", "unit.deviance.adj", "unit.df.adj", "fitted.values"))
    write.table(unname(fit_unit[[m]]), file.path(out, paste0(gsub(".", "_", m, fixed = TRUE), ".tsv")),
                sep = "\t", row.names = FALSE, col.names = FALSE)
  write.table(unname(as.matrix(fit_unit$offset)), file.path(out, "offset.tsv"), sep = "\t",
              row.names = FALSE, col.names = FALSE)
}

scalar <- function(v) if (length(v) == 1L) unname(v) else NULL
write_json(list(
  edger = as.character(packageVersion("edgeR")), limma = as.character(packageVersion("limma")),
  r = R.version.string, coefficient = coef, coefficient_index = j,
  genes = nrow(counts), samples = ncol(counts), kept = sum(keep),
  ql = list(dispersion = scalar(fit$dispersion), top_proportion = fit$top.proportion,
            average_ql_dispersion = fit$average.ql.dispersion, prior_count = fit$prior.count,
            df_prior = scalar(fit$df.prior), s2_prior = scalar(fit$s2.prior), method = fit$method,
            discoveries_fdr_0_1 = sum(ql_tab$FDR < 0.1), discoveries_fdr_0_05 = sum(ql_tab$FDR < 0.05)),
  lrt = list(common_dispersion = yd$common.dispersion, trend_method = yd$trend.method,
             prior_df = yd$prior.df, prior_n = yd$prior.n, span = yd$span,
             discoveries_fdr_0_1 = sum(lrt_tab$FDR < 0.1), discoveries_fdr_0_05 = sum(lrt_tab$FDR < 0.05))
), file.path(out, "reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
write_json(list(threads = threads, blas = extSoftVersion()[["BLAS"]], steps = seconds,
                ql_seconds = ql_seconds, lrt_seconds = lrt_seconds,
                total_seconds = sum(unlist(seconds))),
           file.path(out, "timing.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("ql", ql_seconds, "lrt", lrt_seconds, "kept", sum(keep), "ql_fdr0.1", sum(ql_tab$FDR < 0.1),
    "lrt_fdr0.1", sum(lrt_tab$FDR < 0.1), "\n")
