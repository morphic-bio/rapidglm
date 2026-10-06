# Public API only. No SCEPTRE implementation or internal fitting functions.
args<-commandArgs(TRUE);root<-normalizePath(args[1],mustWork=FALSE);.libPaths(c(args[2],.libPaths()))
suppressPackageStartupMessages(library(sceptre));suppressPackageStartupMessages(library(Matrix));suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('sceptre'))=='0.99.0')
dir.create(root,recursive=TRUE,showWarnings=FALSE)
write_numeric<-function(x,p)write.table(x,p,sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
for(mode in c('low','high')) {
 set.seed(20260926);env<-new.env();name<-paste0(mode,'moi_example_data');data(list=name,package='sceptre',envir=env);d<-env[[name]]
 if(mode=='high'){data('grna_target_data_frame_highmoi',package='sceptre',envir=env);map<-env$grna_target_data_frame_highmoi}else map<-d$grna_target_data_frame
 map<-map[match(rownames(d$grna_matrix),map$grna_id),,drop=FALSE]
 # Fixed calls are supplied identically to both implementations; thresholding is public API.
 calls<-as(d$grna_matrix>=5,'dgCMatrix');grna<-calls*5
 covariates<-data.frame(log_total=log1p(Matrix::colSums(d$response_matrix)),log_detected=log1p(Matrix::colSums(d$response_matrix>0)),batch=d$extra_covariates$batch)
 formula<-~log_total+log_detected+batch
 targets<-head(unique(map$grna_target[map$grna_target!='non-targeting']),6)
 responses<-rownames(d$response_matrix)[head(which(Matrix::rowSums(d$response_matrix>0)>50),12)]
 pairs<-expand.grid(grna_target=targets,response_id=responses,stringsAsFactors=FALSE)
 directory<-file.path(root,paste0(if(mode=='low')'S1' else 'S2'));dir.create(directory,recursive=TRUE,showWarnings=FALSE)
 writeMM(as(Matrix(d$response_matrix,sparse=TRUE),'generalMatrix'),file.path(directory,'responses.mtx'));writeMM(as(calls,'generalMatrix'),file.path(directory,'assignments.mtx'))
 write_numeric(model.matrix(formula,covariates),file.path(directory,'design.tsv'))
 write.table(data.frame(id=map$grna_id,target=map$grna_target,non_targeting=as.integer(map$grna_target=='non-targeting')),file.path(directory,'guides.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 write.table(data.frame(response=match(pairs$response_id,rownames(d$response_matrix))-1,target=pairs$grna_target,kind='discovery'),file.path(directory,'pairs.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 write.table(data.frame(response=seq_len(nrow(d$response_matrix))-1,response_id=rownames(d$response_matrix)),file.path(directory,'response_ids.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 config<-list(responses=file.path(directory,'responses.mtx'),assignments=file.path(directory,'assignments.mtx'),design=file.path(directory,'design.tsv'),guides=file.path(directory,'guides.tsv'),pairs=file.path(directory,'pairs.tsv'),output=file.path(directory,'native'),family='nb',moi=mode,control_group=if(mode=='low')'nt_cells' else 'complement',resampling_mechanism=if(mode=='low')'permutations' else 'crt',cell_qc=FALSE,seed=20260926,budgets=c(499,4999),effect_fits='all',threads=1)
 write_json(config,file.path(directory,'config.json'),auto_unbox=TRUE,pretty=TRUE)
 start<-proc.time()[['elapsed']]
 obj<-import_data(response_matrix=d$response_matrix,grna_matrix=grna,grna_target_data_frame=map,moi=mode,extra_covariates=covariates)
 obj<-set_analysis_parameters(obj,discovery_pairs=pairs,formula_object=formula,control_group=config$control_group,resampling_mechanism=config$resampling_mechanism)
 obj<-assign_grnas(obj,method='thresholding',threshold=5L,print_progress=FALSE)
 obj<-run_qc(obj,response_n_umis_range=c(0,1),response_n_nonzero_range=c(0,1),p_mito_threshold=1)
 obj<-run_discovery_analysis(obj,output_amount=2,print_progress=FALSE)
 elapsed<-proc.time()[['elapsed']]-start
 result<-get_result(obj,'run_discovery_analysis');write.table(result,file.path(directory,'reference.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
 write_json(list(version=as.character(packageVersion('sceptre')),seed=20260926,end_to_end_seconds=elapsed,pairs=nrow(pairs),scope='packaged example subset, fixed threshold-5 assignments; no continuous cell QC trimming'),file.path(directory,'reference.json'),auto_unbox=TRUE,pretty=TRUE)
 cat(mode,elapsed,'seconds\n')
}
