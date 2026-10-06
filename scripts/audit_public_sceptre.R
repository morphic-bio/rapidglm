# Black-box empirical/skew comparisons using documented SCEPTRE 0.99.0 APIs.
args <- commandArgs(TRUE);stopifnot(length(args)==2)
out<-args[1];.libPaths(c(args[2],.libPaths()))
suppressPackageStartupMessages(library(sceptre))
suppressPackageStartupMessages(library(Matrix))
stopifnot(as.character(packageVersion('sceptre'))=='0.99.0')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
e<-new.env();load('validation/local-data/sceptredata/highmoi_example_data.rda',envir=e)
load('validation/local-data/sceptredata/grna_target_data_frame_highmoi.rda',envir=e)
d<-e$highmoi_example_data;map<-e$grna_target_data_frame_highmoi
map<-map[match(rownames(d$grna_matrix),map$grna_id),,drop=FALSE]
grna<-as(d$grna_matrix>=5,'dgCMatrix')*5
covariates<-data.frame(log_total=log1p(colSums(d$response_matrix)),
                       log_detected=log1p(colSums(d$response_matrix>0)),
                       batch=as.factor(d$extra_covariates[[1]]))
original<-read.delim('validation/work/public-sceptre/S2_public/reference.tsv')
pairs<-original[,c('grna_target','response_id')]
rows<-list()
for(approximation in c('skew_normal','no_approximation')) {
  obj<-import_data(d$response_matrix,grna,map,moi='high',extra_covariates=covariates)
  obj<-set_analysis_parameters(obj,discovery_pairs=pairs,formula_object=~log_total+log_detected+batch,
        control_group='complement',resampling_mechanism='crt',resampling_approximation=approximation)
  obj<-assign_grnas(obj,method='thresholding',threshold=5L,print_progress=FALSE)
  obj<-run_qc(obj,response_n_umis_range=c(0,1),response_n_nonzero_range=c(0,1),p_mito_threshold=1)
  for(seed in 20260926:20260928) {
    set.seed(seed);start<-proc.time()[['elapsed']]
    run<-run_discovery_analysis(obj,output_amount=3,print_progress=FALSE)
    seconds<-proc.time()[['elapsed']]-start
    result<-get_result(run,'run_discovery_analysis')
    saveRDS(result,file.path(out,paste0(approximation,'-',seed,'.rds')))
    # Retain simple public result fields in the aggregate table; any detailed
    # list-valued output remains in the per-run RDS.
    result<-as.data.frame(result)
    z<-as.matrix(result[,grepl('^z_null_',names(result)),drop=FALSE])
    draws<-rowSums(is.finite(z))
    empirical<-pmin(1,2*pmin((1+rowSums(z<=result$z_orig,na.rm=TRUE))/(draws+1),
                              (1+rowSums(z>=result$z_orig,na.rm=TRUE))/(draws+1)))
    result<-result[,!grepl('^z_null_',names(result)) &
                    vapply(result,function(x)!is.list(x),logical(1)),drop=FALSE]
    result$approximation<-approximation;result$seed<-seed;result$seconds<-seconds
    result$draws<-draws;result$recomputed_empirical<-empirical
    result$q_value<-p.adjust(result$p_value,'BH')
    rows[[length(rows)+1]]<-result
    write.table(do.call(rbind,rows),file.path(out,'sceptre-repeats.tsv'),
                sep='\t',row.names=FALSE,quote=FALSE)
    message(approximation,' seed ',seed,' complete')
  }
}
