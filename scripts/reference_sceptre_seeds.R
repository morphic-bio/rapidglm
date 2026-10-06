# Black-box SCEPTRE 0.99.0 discovery runs of the public S1/S2 panels under several
# seeds, for the staging validation. Usage: reference_sceptre_seeds.R sceptre_lib out_dir seeds
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); out <- args[2]; seeds <- as.integer(strsplit(args[3], ",")[[1]])
suppressPackageStartupMessages({library(sceptre); library(Matrix)})
dir.create(out, showWarnings = FALSE, recursive = TRUE)
for (mode in c('low', 'high')) {
  e <- new.env(); load(paste0('validation/local-data/sceptredata/', mode, 'moi_example_data.rda'), envir = e); d <- e[[paste0(mode, 'moi_example_data')]]
  if (mode == 'high') { load('validation/local-data/sceptredata/grna_target_data_frame_highmoi.rda', envir = e); map <- e$grna_target_data_frame_highmoi } else map <- d$grna_target_data_frame
  map <- map[match(rownames(d$grna_matrix), map$grna_id), , drop = FALSE]; grna <- as(d$grna_matrix >= 5, 'dgCMatrix') * 5
  covariates <- data.frame(log_total = log1p(Matrix::colSums(d$response_matrix)), log_detected = log1p(Matrix::colSums(d$response_matrix > 0)), batch = as.factor(d$extra_covariates[[1]]))
  panel <- if (mode == 'low') 'S1_public' else 'S2_public'
  ref <- read.delim(file.path('validation/work/public-sceptre', panel, 'reference.tsv'))
  pairs <- unique(ref[, c('grna_target', 'response_id')])
  obj <- import_data(d$response_matrix, grna, map, moi = mode, extra_covariates = covariates)
  obj <- set_analysis_parameters(obj, discovery_pairs = pairs, formula_object = ~log_total + log_detected + batch,
                                 control_group = if (mode == 'low') 'nt_cells' else 'complement',
                                 resampling_mechanism = if (mode == 'low') 'permutations' else 'crt')
  obj <- assign_grnas(obj, method = 'thresholding', threshold = 5L, print_progress = FALSE)
  obj <- run_qc(obj, response_n_umis_range = c(0, 1), response_n_nonzero_range = c(0, 1), p_mito_threshold = 1)
  for (s in seeds) {
    set.seed(s)
    r <- run_discovery_analysis(obj, output_amount = 2, print_progress = FALSE)
    write.table(get_result(r, 'run_discovery_analysis'), file.path(out, paste0(panel, '.seed', s, '.tsv')), sep = '\t', quote = FALSE, row.names = FALSE)
  }
}
