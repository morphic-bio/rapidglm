# Black-box probe of SCEPTRE 0.99.0 staging for the staging validation:
# many calibration (null) pairs on the public high-MOI example (sceptredata), recording
# each pair's stage and p-value. Usage: reference_sceptre_stages.R sceptre_lib out.tsv seed n_pairs
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages({library(sceptre); library(Matrix)})
set.seed(as.integer(args[3])); e <- new.env()
load('validation/local-data/sceptredata/highmoi_example_data.rda', envir = e); d <- e$highmoi_example_data
load('validation/local-data/sceptredata/grna_target_data_frame_highmoi.rda', envir = e); map <- e$grna_target_data_frame_highmoi
map <- map[match(rownames(d$grna_matrix), map$grna_id), , drop = FALSE]
grna <- as(d$grna_matrix >= 5, 'dgCMatrix') * 5
covariates <- data.frame(log_total = log1p(Matrix::colSums(d$response_matrix)), log_detected = log1p(Matrix::colSums(d$response_matrix > 0)), batch = as.factor(d$extra_covariates[[1]]))
obj <- import_data(d$response_matrix, grna, map, moi = 'high', extra_covariates = covariates)
obj <- set_analysis_parameters(obj, formula_object = ~log_total + log_detected + batch, control_group = 'complement', resampling_mechanism = 'crt')
obj <- assign_grnas(obj, method = 'thresholding', threshold = 5L, print_progress = FALSE)
obj <- run_qc(obj, response_n_umis_range = c(0, 1), response_n_nonzero_range = c(0, 1), p_mito_threshold = 1)
obj <- run_calibration_check(obj, n_calibration_pairs = as.integer(args[4]), calibration_group_size = 1L, output_amount = 2, print_progress = FALSE)
write.table(get_result(obj, 'run_calibration_check'), args[2], sep = '\t', quote = FALSE, row.names = FALSE)
