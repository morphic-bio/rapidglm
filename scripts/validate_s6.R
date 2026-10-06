# Execute the existing independent simulator unchanged, replacing only its test callbacks.
# No comparator implementation is copied into this repository or native engine.
args<-commandArgs(TRUE)
stopifnot(length(args)==3)
source_root<-normalizePath(args[1]);out<-normalizePath(args[2],mustWork=FALSE);tier<-args[3]
stopifnot(tier %in% c('smoke','release'))
# Workflow settings under test; 'direct' for both reproduces the receipts made before 2026-09-29.
null_model<-Sys.getenv('RAPIDGLM_S6_NULL_MODEL','sceptre');crt_draw<-Sys.getenv('RAPIDGLM_S6_CRT_DRAW','thinned')
suppressPackageStartupMessages(library(jsonlite));suppressPackageStartupMessages(library(Rcpp));suppressPackageStartupMessages(library(digest))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
profile_path<-file.path(source_root,'profiles/method-evaluation-20260925-v2.json')
cfg<-fromJSON(profile_path,simplifyVector=FALSE);cal<-cfg$calibration
source(file.path(source_root,'R/method_rng.R'))
# Non-system numerical installations can supply the same include/link flags as
# the CMake build. Preserve caller flags; this bridge is a separate R process.
Sys.setenv(PKG_CPPFLAGS=paste(Sys.getenv('PKG_CPPFLAGS'),
                            Sys.getenv('RAPIDGLM_CPPFLAGS'),
                            paste0('-I',shQuote(normalizePath('include')))),
           PKG_LIBS=paste(Sys.getenv('PKG_LIBS'),
                          Sys.getenv('RAPIDGLM_LIBS','-llapacke -llapack -lblas')))
# Header changes must not silently reuse a bridge compiled against an old engine.
sourceCpp('tests/s6_bridge.cpp',cacheDir=file.path(out,'bridge-cache'),rebuild=TRUE)
env<-new.env(parent=globalenv());env$cfg<-cfg;env$cal<-cal;env$base_seed<-if(tier=='release')cal$validation_seed else cal$development_seed
scenarios<-c(unlist(cal$primary_scenarios),unlist(cal$stress_scenarios),'mixed_family')
# Extract the existing generator/test driver as a closure; do not execute its production runner.
expressions<-parse(file.path(source_root,'R/calibrate_spacrt.R'))
assignment<-Filter(function(x)is.call(x)&&identical(x[[1]],as.name('<-'))&&identical(x[[2]],as.name('run_one')),as.list(expressions))
stopifnot(length(assignment)==1);eval(assignment[[1]],envir=env)
env$api<-list(spacrt_roots=function(G,P,residuals,threads){
  frame<-parent.frame();Y<-if(exists('Y',frame,inherits=FALSE))get('Y',frame)else matrix(get('y',frame),ncol=1)
  scenario<-get('scenario',frame);i<-get('i',frame)
  seed<-as.double(env$base_seed)*1000000+match(scenario,scenarios)*100000+i
  # The new workflow estimates NB nulls and logistic propensities in every scenario.
  # Known/fitted describe the original simulator scenarios, not bypassed native fitting.
  rapidglm_s6(Y,cbind(1,get('z',frame)),G[,1],seed,null_model,crt_draw)
})
env$spacrt_native_tails<-function(x)x
env$spacrt_published<-function(...)list(p_value=NA_real_,method='published_spa')
env$spacrt_empirical_v2<-function(...)list(p_value=NA_real_)
datasets<-if(tier=='release')cal$release_datasets else cal$smoke_datasets
method_check_rng_keys(unlist(lapply(scenarios,function(s)paste(env$base_seed,s,seq_len(datasets),sep=':'))))
rows<-list();start<-proc.time()[['elapsed']]
for(scenario in scenarios) {
  parts<-parallel::mclapply(seq_len(datasets),function(i)env$run_one(scenario,i),mc.cores=8,mc.set.seed=FALSE)
  stopifnot(!any(vapply(parts,inherits,logical(1),'try-error')))
  rows[[scenario]]<-do.call(rbind,parts);saveRDS(rows[[scenario]],file.path(out,paste0(scenario,'.rds')))
  message(scenario,' ',datasets,' complete')
}
rows<-do.call(rbind,rows);write.table(rows,file.path(out,'datasets.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
null<-list();ci_alpha<-(1-cal$confidence)/(length(cal$primary_scenarios)*length(cal$alphas))
for(scenario in scenarios[scenarios!='mixed_family'])for(k in seq_along(cal$alphas)) {
 p<-rows$p_value[rows$scenario==scenario];hits<-sum(p<=cal$alphas[[k]],na.rm=TRUE);failures<-sum(!is.finite(p))
 upper<-if(hits==length(p))1 else qbeta(1-ci_alpha,hits+1,length(p)-hits)
 null[[length(null)+1]]<-data.frame(scenario=scenario,alpha=cal$alphas[[k]],datasets=length(p),rejections=hits,failures=failures,upper=upper,limit=cal$upper_bounds[[k]],primary=scenario%in%unlist(cal$primary_scenarios),pass=failures==0 && upper<=cal$upper_bounds[[k]])
}
null<-do.call(rbind,null);fdp<-rows$fdp[rows$scenario=='mixed_family'];family_upper<-if(all(is.finite(fdp)))min(1,mean(fdp)+sqrt(2*var(fdp)*log(3/.05)/length(fdp))+3*log(3/.05)/(length(fdp)-1))else NA_real_
passed<-tier=='release'&&all(null$pass[null$primary])&&is.finite(family_upper)&&family_upper<=cal$family_upper_bound
receipt<-list(gate='G5',status=if(tier=='smoke')'SMOKE_ONLY'else if(passed)'PASS_DECLARED_SIMULATIONS_ONLY'else'FAIL_OR_INCONCLUSIVE',tier=tier,datasets_per_scenario=datasets,null=null,family_mean_fdp=mean(fdp),family_upper=family_upper,family_limit=cal$family_upper_bound,family_power=mean(rows$power[rows$scenario=='mixed_family']),seconds=proc.time()[['elapsed']]-start,source_simulator_sha256=digest(file=file.path(source_root,'R/calibrate_spacrt.R'),algo='sha256'),source_rng_sha256=digest(file=file.path(source_root,'R/method_rng.R'),algo='sha256'),source_profile_sha256=digest(file=profile_path,algo='sha256'),dataset_sha256=digest(file=file.path(out,'datasets.tsv'),algo='sha256'),native_bridge_sha256=digest(file='tests/s6_bridge.cpp',algo='sha256'),seed=env$base_seed,law=paste0('NB fitted response null (null_model=',null_model,') + fitted logistic CRT (crt_draw=',crt_draw,'); staged skew-normal 499/4999; pair QC disabled for test-calibration comparability'),null_model=null_model,crt_draw=crt_draw,simulator='existing read-only v2 run_one closure; original data/RNG unchanged; inference callbacks replaced with RapidGLM workflow',scope='independent-cell calibration; stress rows do not necessarily satisfy the conditional independence null',production_promoted=FALSE)
write_json(receipt,file.path(out,'S6.json'),auto_unbox=TRUE,pretty=TRUE,na='null',digits=16);print(null);print(family_upper)
