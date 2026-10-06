# Full public example data, pinned separately from black-box SCEPTRE 0.99.0.
args<-commandArgs(TRUE);root<-normalizePath(args[1],mustWork=FALSE);.libPaths(c(args[2],.libPaths()));suppressPackageStartupMessages(library(sceptre));suppressPackageStartupMessages(library(Matrix));suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('sceptre'))=='0.99.0');dir.create(root,recursive=TRUE,showWarnings=FALSE)
write_numeric<-function(x,p)write.table(x,p,sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
for(mode in c('low','high')) {
 set.seed(20260926);e<-new.env();load(paste0('validation/local-data/sceptredata/',mode,'moi_example_data.rda'),envir=e);d<-e[[paste0(mode,'moi_example_data')]]
 if(mode=='high'){load('validation/local-data/sceptredata/grna_target_data_frame_highmoi.rda',envir=e);map<-e$grna_target_data_frame_highmoi}else map<-d$grna_target_data_frame
 map<-map[match(rownames(d$grna_matrix),map$grna_id),,drop=FALSE];calls<-as(d$grna_matrix>=5,'dgCMatrix');grna<-calls*5
 covariates<-data.frame(log_total=log1p(Matrix::colSums(d$response_matrix)),log_detected=log1p(Matrix::colSums(d$response_matrix>0)),batch=as.factor(d$extra_covariates[[1]]));formula<-~log_total+log_detected+batch
 directory<-file.path(root,if(mode=='low')'S1_public'else'S2_public');dir.create(directory,recursive=TRUE,showWarnings=FALSE)
 # First six identifiable own-gene targets, followed by additional targets if needed.
 targets<-unique(map$grna_target[map$grna_target!='non-targeting']);own<-intersect(targets,rownames(d$response_matrix));targets<-head(unique(c(own,targets)),6)
 responses<-head(unique(c(own,rownames(d$response_matrix)[Matrix::rowSums(d$response_matrix>0)>50])),12)
 pairs<-expand.grid(grna_target=targets,response_id=responses,stringsAsFactors=FALSE)
 obj<-import_data(d$response_matrix,grna,map,moi=mode,extra_covariates=covariates)
 positives<-construct_positive_control_pairs(obj);if(nrow(positives)>0)pairs<-pairs[!paste(pairs$grna_target,pairs$response_id)%in%paste(positives$grna_target,positives$response_id),,drop=FALSE]
 obj<-set_analysis_parameters(obj,discovery_pairs=pairs,positive_control_pairs=positives,formula_object=formula,control_group=if(mode=='low')'nt_cells'else'complement',resampling_mechanism=if(mode=='low')'permutations'else'crt')
 obj<-assign_grnas(obj,method='thresholding',threshold=5L,print_progress=FALSE);obj<-run_qc(obj,response_n_umis_range=c(0,1),response_n_nonzero_range=c(0,1),p_mito_threshold=1)
 start<-proc.time()[['elapsed']];obj<-run_calibration_check(obj,n_calibration_pairs=72L,calibration_group_size=1L,output_amount=2,print_progress=FALSE);calibration_seconds<-proc.time()[['elapsed']]-start
 calibration<-get_result(obj,'run_calibration_check');write.table(calibration,file.path(directory,'reference_calibration.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 start<-proc.time()[['elapsed']];obj<-run_power_check(obj,output_amount=2,print_progress=FALSE);power_seconds<-proc.time()[['elapsed']]-start
 power<-get_result(obj,'run_power_check');write.table(power,file.path(directory,'reference_power.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 start<-proc.time()[['elapsed']];obj<-run_discovery_analysis(obj,output_amount=2,print_progress=FALSE);discovery_seconds<-proc.time()[['elapsed']]-start
 write.table(get_result(obj,'run_discovery_analysis'),file.path(directory,'reference.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 writeMM(as(Matrix(d$response_matrix,sparse=TRUE),'generalMatrix'),file.path(directory,'responses.mtx'));writeMM(as(calls,'generalMatrix'),file.path(directory,'assignments.mtx'));write_numeric(model.matrix(formula,covariates),file.path(directory,'design.tsv'))
 write.table(data.frame(response=seq_len(nrow(d$response_matrix))-1,response_id=rownames(d$response_matrix)),file.path(directory,'response_ids.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
 # Keep NT guide identities explicit for deterministic, reproducible negative controls.
 nt<-map$grna_target=='non-targeting';native_map<-map;native_map$grna_target<-as.character(native_map$grna_target);native_map$grna_target[nt]<-paste0('NT:',map$grna_id[nt])
 write.table(data.frame(id=map$grna_id,target=native_map$grna_target,non_targeting=as.integer(nt)),file.path(directory,'guides.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
 negative_pairs<-expand.grid(response_id=responses,grna_target=head(native_map$grna_target[nt],6),stringsAsFactors=FALSE)
 allpairs<-rbind(data.frame(pairs,kind='discovery'),data.frame(positives,kind='positive'),data.frame(negative_pairs,kind='negative'))
 write.table(data.frame(response=match(allpairs$response_id,rownames(d$response_matrix))-1,target=allpairs$grna_target,kind=allpairs$kind),file.path(directory,'pairs.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
 config<-list(responses=file.path(directory,'responses.mtx'),assignments=file.path(directory,'assignments.mtx'),design=file.path(directory,'design.tsv'),guides=file.path(directory,'guides.tsv'),pairs=file.path(directory,'pairs.tsv'),output=file.path(directory,'native'),family='nb',moi=mode,control_group=if(mode=='low')'nt_cells'else'complement',resampling_mechanism=if(mode=='low')'permutations'else'crt',cell_qc=FALSE,seed=20260926,budgets=c(499,4999),effect_fits='all',threads=2)
 write_json(config,file.path(directory,'config.json'),auto_unbox=TRUE,pretty=TRUE)
 write_json(list(version=as.character(packageVersion('sceptre')),cells=ncol(d$response_matrix),features=nrow(d$response_matrix),discovery_seconds=discovery_seconds,power_seconds=power_seconds,calibration_seconds=calibration_seconds,scope='full public sceptredata example cells; fixed threshold-5 calls; discovery and positive control panels matched; negative control family separately sampled'),file.path(directory,'reference.json'),auto_unbox=TRUE,pretty=TRUE)
 cat(mode,'complete\n')
}
