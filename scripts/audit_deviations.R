# Diagnostic comparisons through documented R APIs and public RapidGLM APIs.
# Usage: Rscript scripts/audit_deviations.R MODE OUT [simulator_root]
args <- commandArgs(TRUE)
stopifnot(length(args) >= 2)
mode <- args[1]; out <- args[2]
stopifnot(mode %in% c('s6','numerical','atac','pseudobulk','s2','theta','timing','timing16','s2-score','sceptre-model'))
dir.create(out, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(jsonlite))
Sys.setenv(PKG_CPPFLAGS=paste(Sys.getenv('RAPIDGLM_CPPFLAGS'),
                            paste0('-I',shQuote(normalizePath('include')))),
           PKG_LIBS=Sys.getenv('RAPIDGLM_LIBS','-llapacke -llapack -lblas'))
Rcpp::sourceCpp('tests/deviation_bridge.cpp',cacheDir=file.path(out,'bridge-cache'))
read_matrix <- function(p) as.matrix(read.table(p,header=FALSE))
write_tsv <- function(x,p) write.table(x,p,sep='\t',row.names=FALSE,quote=FALSE)
json <- function(x,p) write_json(x,p,auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')

if(mode == 's6') {
  stopifnot(length(args)%in%c(3,5))
  source_root <- normalizePath(args[3])
  cfg <- fromJSON(file.path(source_root,'profiles/method-evaluation-20260925-v2.json'),simplifyVector=FALSE)
  cal <- cfg$calibration
  source(file.path(source_root,'R/method_rng.R'))
  env <- new.env(parent=globalenv())
  env$cfg <- cfg; env$cal <- cal; env$base_seed <- cal$validation_seed
  expressions <- parse(file.path(source_root,'R/calibrate_spacrt.R'))
  assignment <- Filter(function(x)is.call(x)&&identical(x[[1]],as.name('<-'))&&
                         identical(x[[2]],as.name('run_one')),as.list(expressions))
  stopifnot(length(assignment)==1);eval(assignment[[1]],envir=env)
  scenarios <- c(unlist(cal$primary_scenarios),unlist(cal$stress_scenarios),'mixed_family')
  env$api <- list(spacrt_roots=function(G,P,residuals,threads) {
    frame <- parent.frame(); scenario <- get('scenario',frame); i <- get('i',frame)
    seed <- as.double(env$base_seed)*1000000+match(scenario,scenarios)*100000+i
    rows <- audit_tails(get('y',frame),cbind(1,get('z',frame)),G[,1],
                        get('truth_p',frame),seed)
    rows$dataset <- i; rows$scenario <- scenario
    # Stop only this diagnostic invocation after the unchanged simulator has
    # supplied its data. Do not execute unrelated comparator inference callbacks.
    stop(structure(list(message='audit complete',rows=rows),
                   class=c('audit_complete','error','condition')))
  })
  env$spacrt_native_tails <- function(x)x
  env$spacrt_published <- function(...)list(p_value=NA_real_,method='published_spa')
  env$spacrt_empirical_v2 <- function(...)list(p_value=NA_real_)
  n <- if(length(args)==5)as.integer(args[5])else cal$release_datasets
  first <- if(length(args)==5)as.integer(args[4])else 1L
  stopifnot(n>0,first>0)
  start <- proc.time()[['elapsed']]
  # Surface setup/bridge errors directly before starting worker processes.
  stopifnot(nrow(tryCatch(env$run_one('known_dense',1),
                          audit_complete=function(e)e$rows))==6)
  rows <- do.call(rbind,parallel::mclapply(seq.int(first,length.out=n),function(i)
    tryCatch(env$run_one('known_dense',i),audit_complete=function(e)e$rows),
    mc.cores=8,mc.set.seed=FALSE))
  stopifnot(nrow(rows)==n*6)
  write_tsv(rows,file.path(out,'s6-ablation.tsv'))
  summaries <- do.call(rbind,lapply(split(rows,list(rows$law,rows$method)),function(d)
    do.call(rbind,lapply(c(.05,.01),function(alpha) {
      hits <- sum(d$p_value<=alpha)
      data.frame(law=d$law[1],method=d$method[1],alpha=alpha,n=nrow(d),hits=hits,
                 upper=qbeta(1-.05/8,hits+1,nrow(d)-hits),
                 approximations=sum(d$approximate),mean_draws=mean(d$draws))
    }))))
  json(list(scope=if(first==1)'paired diagnostic of original failed known_dense panel; thresholds remain provisional'else
                  paste('fixed',n,'dataset diagnostic extension, independently keyed datasets; thresholds remain provisional'),
            seed=env$base_seed,first_dataset=first,datasets=n,
            seconds=proc.time()[['elapsed']]-start,results=summaries),
       file.path(out,'s6-ablation.json'))
  print(summaries)
}

if(mode == 'numerical') {
  results <- list(); times <- list()
  for(panel in c('E1_nb','S2_public_nb')) {
    d <- file.path('validation/work/glm',panel)
    x <- read_matrix(file.path(d,'design.tsv'))
    Y <- t(as.matrix(readMM(file.path(d,'responses.mtx'))))
    off <- scan(file.path(d,'offset.tsv'),quiet=TRUE)
    w <- scan(file.path(d,'weights.tsv'),quiet=TRUE)
    keep <- w>0; x<-x[keep,,drop=FALSE];Y<-Y[keep,,drop=FALSE];off<-off[keep];w<-w[keep]
    for(tolerance in c(1e-9,1e-11,1e-13)) {
      start <- proc.time()[['elapsed']]
      native <- audit_fits(x,Y,off,w,numeric(),tolerance=tolerance)
      native_seconds <- proc.time()[['elapsed']]-start
      start <- proc.time()[['elapsed']]
      reference <- lapply(seq_len(ncol(Y)),function(j)
        glm.nb(Y[,j]~x-1+offset(off),weights=w,
               control=glm.control(epsilon=tolerance,maxit=1000)))
      r_seconds <- proc.time()[['elapsed']]-start
      times[[length(times)+1]] <- data.frame(panel=panel,tolerance=tolerance,
                                            native_seconds=native_seconds,r_seconds=r_seconds,
                                            speedup=r_seconds/native_seconds)
      for(j in seq_along(reference)) {
        f<-reference[[j]]; mu<-f$fitted.values
        final_se <- sqrt(diag(solve(crossprod(x,x*as.vector(w*mu/(1+mu/f$theta))))))
        results[[length(results)+1]] <- data.frame(panel=panel,feature=j-1,tolerance=tolerance,
           native_theta=native$theta[j],r_theta=f$theta,
           theta_abs=abs(native$theta[j]-f$theta),
           beta_max_abs=max(abs(native$beta[j,]-coef(f))),
           se_max_abs=max(abs(native$se[j,]-final_se)),
           loglik_difference=native$loglik[j]-as.numeric(logLik(f)),status=native$status[j])
      }
      saveRDS(list(native=native,reference=reference),
              file.path(out,paste0(panel,'-',tolerance,'.rds')))
    }
  }
  write_tsv(do.call(rbind,results),file.path(out,'numerical.tsv'))
  write_tsv(do.call(rbind,times),file.path(out,'fit-timings.tsv'))
  print(do.call(rbind,times))
}

if(mode == 'atac') {
  d <- 'validation/work/multimodal'
  G <- readMM(file.path(d,'assignments.mtx'))
  guides <- read.delim(file.path(d,'guides.tsv'))
  Y <- readMM(file.path(d,'atac/responses.mtx'))
  X <- read_matrix(file.path(d,'atac/design.tsv'))
  comparison <- read.delim(file.path(d,'atac/direction_comparison.tsv'))
  eligible <- colSums(G)==1
  nt <- colSums(G[guides$non_targeting==1,,drop=FALSE])>0
  rows <- lapply(seq_len(nrow(comparison)),function(j) {
    g <- colSums(G[guides$target==comparison$target[j],,drop=FALSE])>0
    keep <- eligible & (g | nt)
    x <- cbind(X[keep,,drop=FALSE],g[keep]*1)
    y <- as.numeric(Y[comparison$response[j]+1,keep])
    if(sum(y[g[keep]]>0)<7 || sum(y[!g[keep]]>0)<7) return(NULL)
    f <- glm.nb(y~x-1,control=glm.control(epsilon=1e-12,maxit=1000))
    native <- audit_fits(x,matrix(y,ncol=1),numeric(),numeric(),numeric(),tolerance=1e-12)
    data.frame(target=comparison$target[j],cells=sum(keep),target_cells=sum(g&keep),
       nt_cells=sum(nt&keep),native_log2fc=native$beta[1,3]/log(2),
       r_log2fc=coef(f)[3]/log(2),historical_log2fc=comparison$log2fc[j],
       raw_log2_ratio=log2(mean(y[g[keep]])/mean(y[!g[keep]])),
       depth_normalized_log2_ratio=log2(mean(y[g[keep]]/expm1(x[g[keep],2]))/
                                         mean(y[!g[keep]]/expm1(x[!g[keep],2]))),
       paper_delta=comparison$paper_delta[j],suite_delta=comparison$suite_delta[j])
  })
  rows <- do.call(rbind,rows);write_tsv(rows,file.path(out,'atac-matched.tsv'));print(rows)
}

if(mode == 'pseudobulk') {
  .libPaths(c('validation/local-data/R_library',.libPaths()))
  suppressPackageStartupMessages(library(edgeR))
  suppressPackageStartupMessages(library(DESeq2))
  d <- 'validation/work/applications/pseudobulk'
  X <- read_matrix(file.path(d,'design.tsv'))
  Y <- as.matrix(readMM(file.path(d,'responses.mtx')))
  off <- scan(file.path(d,'offset.tsv'),quiet=TRUE)
  original <- read.delim(file.path(d,'reference.tsv'))
  edge <- estimateDisp(DGEList(Y,lib.size=exp(off)),X)
  ef <- glmFit(edge,X,prior.count=0)
  er <- glmLRT(ef,coef=ncol(X))
  en <- audit_fits(X,t(Y),off,numeric(),1/ef$dispersion,tolerance=1e-12)
  enull <- audit_fits(X[,-ncol(X)],t(Y),off,numeric(),1/ef$dispersion,tolerance=1e-12)
  # The raw likelihood comparison is useful diagnostically, but must not bypass
  # the production API's boundary guard for ordinary chi-square LR inference.
  diagnostic_lr <- pchisq(pmax(0,2*(en$loglik-enull$loglik)),1,lower.tail=FALSE)
  native_lr <- diagnostic_lr
  native_lr[en$status!='converged' | enull$status!='converged'] <- NA_real_
  # DESeq2's public interface accepts the same full-rank model matrix.
  dds <- DESeqDataSetFromMatrix(round(Y),data.frame(row.names=as.character(seq_len(ncol(Y)))),X)
  sizeFactors(dds) <- exp(off-mean(off))
  dds <- DESeq(dds,quiet=TRUE)
  dr <- results(dds,independentFiltering=FALSE,cooksCutoff=FALSE)
  dn <- audit_fits(X,t(Y),off,numeric(),1/dispersions(dds),tolerance=1e-12)
  tight <- nbinomWaldTest(dds,betaPrior=FALSE,betaTol=1e-12,maxit=1000,minmu=1e-8,
                         modelMatrix=X,quiet=TRUE)
  tight_result <- results(tight,independentFiltering=FALSE,cooksCutoff=FALSE)
  rows <- data.frame(feature=seq_len(nrow(Y))-1,edge_theta=1/ef$dispersion,
     edge_beta=ef$coefficients[,ncol(X)],native_edge_beta=en$beta[,ncol(X)],
     edge_p=er$table$PValue,native_edge_lr_p=native_lr,
     diagnostic_edge_lr_p=diagnostic_lr,
     edge_original_p=original$edgeR_p,
     deseq_theta=1/dispersions(dds),deseq_beta=dr$log2FoldChange*log(2),
     native_deseq_beta=dn$beta[,ncol(X)],deseq_p=dr$pvalue,native_deseq_wald_p=dn$p,
     deseq_tight_beta=tight_result$log2FoldChange*log(2),deseq_tight_p=tight_result$pvalue,
     deseq_original_p=original$DESeq2_p,edge_status=en$status,deseq_status=dn$status)
  write_tsv(rows,file.path(out,'pseudobulk-matched.tsv'))
  saveRDS(list(edge=ef,deseq=dds,native_edge=en,native_deseq=dn),
          file.path(out,'pseudobulk-matched.rds'))
  print(sapply(rows[,c('edge_p','native_edge_lr_p','deseq_p','native_deseq_wald_p')],
               function(p)sum(p.adjust(replace(p,!is.finite(p),1),'BH')<=.05)))
}

if(mode == 's2') {
  d <- 'validation/work/public-sceptre/S2_public'
  X <- read_matrix(file.path(d,'design.tsv'))
  Y <- readMM(file.path(d,'responses.mtx')); G <- readMM(file.path(d,'assignments.mtx'))
  guides <- read.delim(file.path(d,'guides.tsv')); pairs <- read.delim(file.path(d,'pairs.tsv'))
  ids <- read.delim(file.path(d,'response_ids.tsv'))
  rows <- parallel::mclapply(which(pairs$kind=='discovery'),function(i) {
    g <- as.numeric(colSums(G[guides$target==pairs$target[i],,drop=FALSE])>0)
    row <- audit_tails(as.numeric(Y[pairs$response[i]+1,]),X,g,numeric(),20260926,i-1)
    row$target <- pairs$target[i];row$response <- pairs$response[i]
    row$response_id <- ids$response_id[match(pairs$response[i],ids$response)]
    row
  },mc.cores=8,mc.set.seed=FALSE)
  stopifnot(!any(vapply(rows,inherits,logical(1),'try-error')))
  rows <- do.call(rbind,rows); rows$q_value <- NA_real_
  for(method in unique(rows$method)) {
    keep <- rows$method==method;rows$q_value[keep] <- p.adjust(rows$p_value[keep],'BH')
  }
  write_tsv(rows,file.path(out,'s2-tails.tsv'))
  print(aggregate(q_value~method,rows,function(q)sum(q<=.1)))
}

if(mode %in% c('theta','timing')) {
  rows <- list()
  for(panel in c('E1_nb','S2_public_nb')) {
    d <- file.path('validation/work/glm',panel)
    X <- read_matrix(file.path(d,'design.tsv'))
    Y <- t(as.matrix(readMM(file.path(d,'responses.mtx'))))
    off <- scan(file.path(d,'offset.tsv'),quiet=TRUE)
    w <- scan(file.path(d,'weights.tsv'),quiet=TRUE)
    keep <- w>0;X<-X[keep,,drop=FALSE];Y<-Y[keep,,drop=FALSE];off<-off[keep];w<-w[keep]
    if(mode=='timing') {
      # Same 12 response columns, full NB ML fits, loaded inputs, one thread.
      # The zero repetition warms both paths and is not included in the summary.
      for(repetition in 0:3) {
        start<-proc.time()[['elapsed']]
        native<-audit_fits(X,Y,off,w,numeric(),tolerance=1e-9)
        native_seconds<-proc.time()[['elapsed']]-start
        start<-proc.time()[['elapsed']]
        reference<-lapply(seq_len(ncol(Y)),function(j)
          glm.nb(Y[,j]~X-1+offset(off),weights=w,
                 control=glm.control(epsilon=1e-9,maxit=1000)))
        r_seconds<-proc.time()[['elapsed']]-start
        rows[[length(rows)+1]]<-data.frame(panel=panel,repetition=repetition,
              native_seconds=native_seconds,r_seconds=r_seconds)
      }
    } else {
      j<-if(panel=='E1_nb')1 else 10;y<-Y[,j]
      original<-glm.nb(y~X-1+offset(off),weights=w,
                        control=glm.control(epsilon=1e-13,maxit=1000))
      native<-audit_fits(X,matrix(y,ncol=1),off,w,numeric(),tolerance=1e-13)
      # Differentiate the documented NB log-likelihood directly, then use R's
      # generic root solver. This does not inspect MASS's optimizer implementation.
      score<-function(theta,mu)sum(w*(digamma(y+theta)-digamma(theta)+log(theta)-
                              log(theta+mu)+1-(y+theta)/(theta+mu)))
      beta<-coef(original);theta<-original$theta
      for(iteration in 1:30) {
        mu<-exp(drop(X%*%beta)+off)
        next_theta<-uniroot(function(t)score(t,mu),c(theta/2,theta*2),tol=1e-14)$root
        f<-glm(y~X-1+offset(off),weights=w,start=beta,family=negative.binomial(next_theta),
               control=glm.control(epsilon=1e-13,maxit=1000))
        change<-max(abs(coef(f)-beta),abs(next_theta-theta))
        beta<-coef(f);theta<-next_theta
        if(change<1e-13)break
      }
      native_mu<-exp(drop(X%*%native$beta[1,])+off)
      rows[[length(rows)+1]]<-data.frame(panel=panel,feature=j-1,
         original_theta=original$theta,native_theta=native$theta,refined_theta=theta,
         original_theta_score=score(original$theta,original$fitted.values),
         native_theta_score=score(native$theta,native_mu),refined_theta_score=score(theta,f$fitted.values),
         refined_beta_max_abs=max(abs(beta-native$beta[1,])),
         mass_explicit_tight_theta=as.numeric(theta.ml(y,native_mu,weights=w,limit=100,eps=1e-12)),
         refinement_iterations=iteration)
    }
  }
  rows<-do.call(rbind,rows);write_tsv(rows,file.path(out,paste0(mode,'-refined.tsv')));print(rows)
}

if(mode=='s2-score') {
  d<-'validation/work/public-sceptre/S2_public'
  X<-read_matrix(file.path(d,'design.tsv'));Y<-readMM(file.path(d,'responses.mtx'))
  G<-readMM(file.path(d,'assignments.mtx'));guides<-read.delim(file.path(d,'guides.tsv'))
  ids<-read.delim(file.path(d,'response_ids.tsv'));reference<-read.delim(file.path(d,'reference.tsv'))
  rows<-list()
  for(response in unique(reference$response_id)) {
    y<-as.numeric(Y[match(response,ids$response_id),])
    f<-glm(y~X-1,family=poisson(),control=glm.control(epsilon=1e-12,maxit=200))
    mu<-f$fitted.values
    moment<-sum(mu^2)/sum((y-mu)^2-mu)
    mle<-as.numeric(theta.ml(y,mu,limit=100,eps=1e-12))
    for(i in which(reference$response_id==response)) {
      g<-as.numeric(colSums(G[guides$target==reference$grna_target[i],,drop=FALSE])>0)
      ratio<-sum(y*g)/sum(mu*g)
      for(law in c('moment','conditional_ml')) {
        theta<-if(law=='moment')moment else mle
        w<-mu/(1+mu/theta);s<-(y-mu)/(1+mu/theta)
        c<-crossprod(X,w*g);v<-sum(w*g)-sum(c*solve(crossprod(X,w*X),c))
        z<-sum(g*s)/sqrt(v)
        rows[[length(rows)+1]]<-data.frame(response_id=response,target=reference$grna_target[i],
            law=law,theta=theta,z=z,r_z=reference$z_orig[i],ratio=ratio,r_ratio=reference$fold_change[i])
      }
    }
  }
  rows<-do.call(rbind,rows);write_tsv(rows,file.path(out,'s2-score.tsv'))
  print(aggregate(cbind(z_error=abs(z-r_z),ratio_error=abs(ratio-r_ratio))~law,rows,max))
}

if(mode=='timing16') {
  stopifnot(audit_workers(16L)==16L,Sys.getenv('OPENBLAS_NUM_THREADS')=='1')
  rows<-list();checks<-list()
  for(panel in c('E1_nb','S2_public_nb','S2_public_64genes')) {
    if(panel=='S2_public_64genes') {
      d<-'validation/work/public-sceptre/S2_public'
      X<-read_matrix(file.path(d,'design.tsv'))
      counts<-readMM(file.path(d,'responses.mtx'))
      selected<-head(which(rowSums(counts>0)>50),64)
      stopifnot(length(selected)==64)
      Y<-t(as.matrix(counts[selected,,drop=FALSE]));off<-rep(0,nrow(X));w<-rep(1,nrow(X))
      write_tsv(data.frame(feature=selected-1),file.path(out,'timing16-public-features.tsv'))
    } else {
      d<-file.path('validation/work/glm',panel)
      X<-read_matrix(file.path(d,'design.tsv'))
      Y<-t(as.matrix(readMM(file.path(d,'responses.mtx'))))
      off<-scan(file.path(d,'offset.tsv'),quiet=TRUE);w<-scan(file.path(d,'weights.tsv'),quiet=TRUE)
      keep<-w>0;X<-X[keep,,drop=FALSE];Y<-Y[keep,,drop=FALSE];off<-off[keep];w<-w[keep]
    }
    baseline<-NULL
    # No other benchmarks are launched concurrently. Repetition zero warms the
    # full workload; vary native thread order across repetitions.
    for(repetition in 0:3) {
      for(threads in if(repetition%%2==0)c(1L,16L)else c(16L,1L)) {
        load<-readLines('/proc/loadavg',n=1)
        start<-proc.time()[['elapsed']]
        native<-audit_fits(X,Y,off,w,numeric(),tolerance=1e-9,threads=threads)
        seconds<-proc.time()[['elapsed']]-start
        if(is.null(baseline))baseline<-native
        stopifnot(identical(native,baseline))
        rows[[length(rows)+1]]<-data.frame(panel=panel,features=ncol(Y),observations=nrow(Y),
           implementation='native',threads=threads,repetition=repetition,seconds=seconds,load_before=load)
      }
      load<-readLines('/proc/loadavg',n=1)
      start<-proc.time()[['elapsed']]
      reference<-lapply(seq_len(ncol(Y)),function(j)
        glm.nb(Y[,j]~X-1+offset(off),weights=w,control=glm.control(epsilon=1e-9,maxit=1000)))
      seconds<-proc.time()[['elapsed']]-start
      rows[[length(rows)+1]]<-data.frame(panel=panel,features=ncol(Y),observations=nrow(Y),
         implementation='R_MASS',threads=1L,repetition=repetition,seconds=seconds,load_before=load)
      write_tsv(do.call(rbind,rows),file.path(out,'timing16.tsv'))
      message(panel,' repetition ',repetition,' complete')
    }
    checks[[panel]]<-list(features=ncol(Y),observations=nrow(Y),native_statuses=as.list(table(baseline$status)),
       r_converged=sum(vapply(reference,function(f)isTRUE(f$converged),logical(1))),
       native_all_repetitions_and_thread_counts_identical=TRUE,
       max_beta_abs=max(abs(baseline$beta-do.call(rbind,lapply(reference,coef)))),
       max_loglik_abs=max(abs(baseline$loglik-vapply(reference,function(f)as.numeric(logLik(f)),numeric(1)))))
    saveRDS(list(native=baseline,reference=reference),file.path(out,paste0('timing16-',panel,'.rds')))
    json(checks,file.path(out,'timing16-checks.json'))
  }
}

if(mode=='sceptre-model') {
  d<-'validation/work/public-sceptre/S2_public'
  X<-read_matrix(file.path(d,'design.tsv'));Y<-readMM(file.path(d,'responses.mtx'))
  G<-readMM(file.path(d,'assignments.mtx'));guides<-read.delim(file.path(d,'guides.tsv'))
  ids<-read.delim(file.path(d,'response_ids.tsv'));reference<-read.delim(file.path(d,'reference.tsv'))
  fit_rows<-list();score_rows<-list();timing_rows<-list()
  for(response in unique(reference$response_id)) {
    y<-as.numeric(Y[match(response,ids$response_id),])
    indices<-which(reference$response_id==response)
    assignments<-vapply(indices,function(i)
      as.numeric(colSums(G[guides$target==reference$grna_target[i],,drop=FALSE])>0),numeric(nrow(X)))
    poisson<-glm(y~X-1,family=poisson(),control=glm.control(epsilon=1e-12,maxit=200))
    mu<-poisson$fitted.values
    theta<-as.numeric(theta.ml(y,mu,limit=100,eps=1e-12))
    native<-audit_null_model(y,X,assignments,TRUE)
    direct<-audit_null_model(y,X,assignments,FALSE)
    fit_rows[[length(fit_rows)+1]]<-data.frame(response_id=response,status=native$status,
      theta_native=native$theta,theta_R=theta,beta_max_abs=max(abs(native$beta-coef(poisson))),
      mu_max_abs=max(abs(native$mu-mu)),loglik_native=native$loglik,
      loglik_R=sum(dnbinom(y,mu=mu,size=theta,log=TRUE)))
    w<-mu/(1+mu/theta);s<-(y-mu)/(1+mu/theta)
    for(j in seq_along(indices)) {
      i<-indices[j];g<-assignments[,j];c<-crossprod(X,w*g)
      z<-sum(g*s)/sqrt(sum(w*g)-sum(c*solve(crossprod(X,w*X),c)))
      score_rows[[length(score_rows)+1]]<-data.frame(response_id=response,target=reference$grna_target[i],
        native=native$statistic[j],binary=native$binary_statistic[j],R_recipe=z,
        sceptre_reference=reference$z_orig[i],direct=direct$statistic[j],
        native_ratio=native$ratio[j],reference_ratio=reference$fold_change[i])
    }
    # Models and inputs are loaded; repetition zero warms each code path.
    for(repetition in 0:3) for(compatibility in if(repetition%%2==0)c(FALSE,TRUE)else c(TRUE,FALSE)) {
      start<-proc.time()[['elapsed']]
      check<-audit_null_model(y,X,assignments,compatibility)
      seconds<-proc.time()[['elapsed']]-start
      stopifnot(identical(check,if(compatibility)native else direct))
      timing_rows[[length(timing_rows)+1]]<-data.frame(response_id=response,repetition=repetition,
        mode=if(compatibility)'sceptre'else'direct',seconds=seconds)
    }
    message(response,' complete')
  }
  fits<-do.call(rbind,fit_rows);scores<-do.call(rbind,score_rows);timings<-do.call(rbind,timing_rows)
  write_tsv(fits,file.path(out,'fits.tsv'));write_tsv(scores,file.path(out,'scores.tsv'))
  write_tsv(timings,file.path(out,'fit-timing.tsv'))
  json(list(responses=nrow(fits),reference_pairs=nrow(scores),
    all_converged=all(fits$status=='converged'),
    beta_R_max_abs=max(fits$beta_max_abs),mu_R_max_abs=max(fits$mu_max_abs),
    theta_R_max_abs=max(abs(fits$theta_native-fits$theta_R)),
    loglik_R_max_abs=max(abs(fits$loglik_native-fits$loglik_R)),
    score_R_recipe_max_abs=max(abs(scores$native-scores$R_recipe)),
    binary_score_R_recipe_max_abs=max(abs(scores$binary-scores$R_recipe)),
    score_sceptre_max_abs=max(abs(scores$native-scores$sceptre_reference)),
    direct_score_sceptre_max_abs=max(abs(scores$direct-scores$sceptre_reference)),
    mean_ratio_sceptre_max_abs=max(abs(scores$native_ratio-scores$reference_ratio))),
    file.path(out,'checks.json'))
}
