# Public DESeq2 calls only. Arguments match scripts/reference_bulk.R.
# Run from the repository root; the reference script leaves its public outputs in scope.
source("scripts/reference_bulk.R")
mu0 <- assays(dds)[["mu"]]
hat0 <- assays(dds)[["H"]]
cook0 <- assays(dds)[["cooks"]]
flag <- as.logical(mcols(dds)$replace)
flag[is.na(flag)] <- FALSE
alpha0 <- ((counts-mu0)^2*hat0/(ncol(X)*cook0*(1-hat0)^2)-mu0)/mu0^2
valid <- is.finite(alpha0) & is.finite(hat0) & is.finite(cook0) & cook0 > 0 & mu0 > 0 & hat0 > 0 & hat0 < 1
valid[flag, ] <- FALSE
summarize <- function(mask) {
  ii <- which(rowSums(mask) >= 2)
  relative <- absolute <- numeric(length(ii))
  for (k in seq_along(ii)) {
    a <- alpha0[ii[k], mask[ii[k], ]]
    absolute[k] <- diff(range(a))
    relative[k] <- absolute[k]/max(abs(median(a)), 1e-300)
  }
  list(genes = length(ii), sample_values = sum(mask[ii, , drop=FALSE]),
       median_relative_range = unname(median(relative)),
       q95_relative_range = unname(quantile(relative, .95)),
       max_relative_range = max(relative), max_absolute_range = max(absolute),
       all_relative_ranges_at_most_1e_12 = max(relative) <= 1e-12)
}
result <- list(reference_version = as.character(packageVersion("DESeq2")),
               method = "Invert public cooks, H and mu assays directly in R, without a decimal TSV round trip; exclude genes with the public replace flag.",
               replacement_flagged_genes_excluded = sum(flag),
               all_eligible = summarize(valid),
               all_means_at_least_0_5 = summarize(valid & apply(mu0 >= .5, 1, all)))
jsonlite::write_json(result, file.path(out, "inversion_in_memory.json"), auto_unbox=TRUE, digits=NA, pretty=TRUE)
print(result)
