# Black-box DESeq2, edgeR and glmGamPoi tests of contrasts and joint hypotheses for
# scripts/validate_hypotheses.py. Only public functions are called.
# Usage:
#   reference_hypotheses.R deseq2 dataset_dir contrast|lrt spec out.tsv libs
#     dataset_dir holds counts.mtx (genes x samples) and design.tsv (with header)
#   reference_hypotheses.R ql panel_dir contrast|lrt spec out.tsv libs
#     panel_dir holds responses.mtx, design.tsv (no header) and offset.tsv
# spec: comma-separated contrast weights, or zero-based design columns to drop.
args <- commandArgs(TRUE)
.libPaths(c(normalizePath(strsplit(args[6], ":")[[1]]), .libPaths()))
suppressPackageStartupMessages(library(Matrix))
tool <- args[1]; dir <- args[2]; kind <- args[3]; out <- args[5]
spec <- as.numeric(strsplit(args[4], ",")[[1]])
if (tool == "deseq2") {
  suppressPackageStartupMessages(library(DESeq2))
  counts <- as.matrix(readMM(file.path(dir, "counts.mtx"))); storage.mode(counts) <- "integer"
  X <- as.matrix(read.delim(file.path(dir, "design.tsv"), check.names = FALSE))
  colnames(X)[colnames(X) %in% c("intercept", "Intercept")] <- "(Intercept)"
  colnames(counts) <- paste0("s", seq_len(ncol(counts)))
  dds <- DESeqDataSetFromMatrix(counts, data.frame(row.names = colnames(counts), one = rep(1, ncol(counts))), ~1)
  if (kind == "contrast") {
    dds <- DESeq(dds, full = X, betaPrior = FALSE, quiet = TRUE)
    res <- results(dds, contrast = spec)
  } else {
    dds <- DESeq(dds, test = "LRT", full = X, reduced = X[, -(spec + 1), drop = FALSE], quiet = TRUE)
    res <- results(dds)
  }
  write.table(data.frame(gene = seq_len(nrow(dds)) - 1, baseMean = res$baseMean,
                         log2FoldChange = res$log2FoldChange, lfcSE = res$lfcSE, stat = res$stat,
                         pvalue = res$pvalue, padj = res$padj),
              out, sep = "\t", row.names = FALSE, quote = FALSE)
} else {
  suppressPackageStartupMessages(library(edgeR))
  counts <- as.matrix(readMM(file.path(dir, "responses.mtx")))
  design <- as.matrix(read.table(file.path(dir, "design.tsv")))
  colnames(design) <- paste0("c", seq_len(ncol(design)) - 1)
  offset <- scan(file.path(dir, "offset.tsv"), quiet = TRUE)
  y <- estimateDisp(DGEList(counts, lib.size = exp(offset)), design)
  fit <- glmQLFit(y, design)
  ql <- if (kind == "contrast") glmQLFTest(fit, contrast = spec) else glmQLFTest(fit, coef = spec + 1)
  result <- data.frame(feature = seq_len(nrow(counts)) - 1, edgeR_ql = ql$table$PValue)
  gp <- glmGamPoi::glm_gp(counts, design = design, offset = offset, size_factors = FALSE)
  test <- if (kind == "contrast") {
    # A contrast expression over the column names, e.g. c3 - c5.
    terms <- paste0(ifelse(spec[spec != 0] < 0, "-", "+"), abs(spec[spec != 0]), "*",
                    colnames(design)[spec != 0], collapse = " ")
    glmGamPoi::test_de(gp, contrast = !!rlang::parse_expr(terms))
  } else {
    glmGamPoi::test_de(gp, reduced_design = design[, -(spec + 1), drop = FALSE])
  }
  result$glmGamPoi_ql <- test$pval
  write.table(result, out, sep = "\t", row.names = FALSE, quote = FALSE)
}
