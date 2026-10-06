# Paired diagnostic through documented R APIs and outputs. No package source
# inspection, copied implementation, or changes to the native fitting algorithm.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
out <- normalizePath(args[1], mustWork=FALSE)
pseudobulk <- normalizePath(args[2])
seed <- as.integer(args[3])
dir.create(out, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(MASS))
options(digits=17)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)

write_numeric <- function(x, path) {
  # Round-trip doubles rather than relying on display precision.
  write.table(format(as.matrix(x), digits=17, scientific=TRUE, trim=TRUE), path,
              sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)
}
read_matrix <- function(path) as.matrix(read.table(path, header=FALSE))
X <- read_matrix(file.path(pseudobulk, "design.tsv"))
O <- scan(file.path(pseudobulk, "offset.tsv"), quiet=TRUE)
genes <- read.delim(file.path(pseudobulk, "genes.tsv"))
counts <- readMM(file.path(pseudobulk, "responses.mtx"))
boundary <- 5L  # Fifth design column (zero-based 4 in the native API).
zero_rows <- which(X[,boundary] == 1)
stopifnot(length(zero_rows)==2L)
cases <- list()
add_case <- function(name, y, offset, origin) {
  d <- file.path(out, name);dir.create(d, showWarnings=FALSE)
  start <- c(log(mean(y))-mean(offset), rep(0,ncol(X)-1))
  arrays <- list(design=X, response=y, offset=offset, weights=rep(1,nrow(X)), start=start)
  paths <- lapply(names(arrays), function(k) file.path(d,paste0(k,".tsv")))
  names(paths) <- names(arrays)
  binary <- lapply(paths,function(p)sub("tsv$","f64le",p))
  for(k in names(arrays)) {
    write_numeric(arrays[[k]],paths[[k]])
    stopifnot(identical(as.numeric(read_matrix(paths[[k]])),as.numeric(arrays[[k]])))
    writeBin(as.double(arrays[[k]]),binary[[k]],size=8,endian="little")
  }
  cases[[length(cases)+1L]] <<- c(list(name=name,directory=d,origin=origin,
                                     boundary_column=boundary-1L,observations=nrow(X),
                                     parameters=ncol(X),binary_inputs=binary),paths)
}
for (gene in c("NME7","MRPS25")) {
  selected <- which(genes$gene==gene);stopifnot(length(selected)==1L)
  y <- as.numeric(counts[selected,]);stopifnot(all(y[zero_rows]==0))
  add_case(gene,y,O,"Unchanged public pseudobulk data")
  y[zero_rows[1]] <- 1
  add_case(paste0(gene,"_positive_control"),y,O,
           "Diagnostic perturbation: one count added to previously zero donor")
}
# Generate once with this exact R RNG/seed; both fitters read the resulting file.
synthetic <- rpois(nrow(X),exp(1.5+0.3*X[,ncol(X)]))
synthetic[zero_rows] <- 0
add_case("seeded_boundary",synthetic,rep(0,nrow(X)),
         "R rpois, then deliberately set this donor to zero")
synthetic[zero_rows] <- c(1,2)
add_case("seeded_positive_control",synthetic,rep(0,nrow(X)),
         "Same synthetic draw; replace zero donor counts with 1,2")
budgets <- c(seq_len(40),50,60,80,100,200,400,600,700,800,1000)
manifest <- list(seed=seed,rng=RNGkind(),tolerance=1e-9,fixed_theta=10,
                 prefix_budgets=budgets,cases=cases,
                 native_outer_iteration_limit=60,
                 reference_outer_iteration_limit="glm.nb uses the same maxit as its inner fit")
write_json(manifest,file.path(out,"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)

fit_one <- function(x,y,offset,weights,start,model,maxit,trace=FALSE) {
  # Fits themselves should not consume RNG. Check every invocation, including errors.
  set.seed(seed);before <- .Random.seed
  warnings <- messages <- character(); error <- NULL; fitted <- NULL
  output <- capture.output({
    fitted <- tryCatch(withCallingHandlers({
      control <- glm.control(epsilon=1e-9,maxit=maxit,
                             trace=if(trace && model=="nb_ml")3 else trace)
      if(model=="nb_ml")
        glm.nb(y~x-1+offset(offset),weights=weights,start=start,control=control)
      else
        glm.fit(x,y,offset=offset,weights=weights,start=start,
                family=if(model=="poisson")poisson()else negative.binomial(10),control=control)
    },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")},
      message=function(m){messages<<-c(messages,conditionMessage(m));invokeRestart("muffleMessage")}),
      error=function(e){error<<-conditionMessage(e);NULL})
  })
  stopifnot(identical(before,.Random.seed))
  list(fit=fitted,warnings=warnings,messages=messages,error=error,output=output,rng_unchanged=TRUE)
}
get_row <- function(result,model,initialization,maxit,x,y,offset,weights) {
  f <- result$fit
  beta <- if(is.null(f))rep(NA_real_,ncol(x))else as.numeric(coef(f))
  mu <- if(is.null(f))rep(NA_real_,length(y))else f$fitted.values
  theta <- if(model=="nb_fixed")10 else if(model=="nb_ml" && !is.null(f))f$theta else NA_real_
  ll <- if(is.null(f))NA_real_ else if(model=="poisson")sum(weights*dpois(y,mu,log=TRUE))else sum(weights*dnbinom(y,mu=mu,size=theta,log=TRUE))
  se <- NA_real_
  if(!is.null(f) && !anyNA(beta)) {
    information_weights <- weights*mu/(1+if(is.finite(theta))mu/theta else 0)
    covariance <- tryCatch(solve(crossprod(x,information_weights*x)),error=function(e)NULL)
    if(!is.null(covariance))se<-sqrt(covariance[boundary,boundary])
  }
  r <- data.frame(model=model,start=initialization,maxit=maxit,
                  iterations=if(is.null(f))NA_integer_ else f$iter,
                  status=if(is.null(f))"error" else if(isTRUE(f$converged))"converged" else "iteration_limit",
                  theta=theta,log_likelihood=ll,
                  deviance=if(is.null(f))NA_real_ else f$deviance,
                  boundary_beta=beta[boundary],boundary_se=se,boundary_mu=sum(mu[zero_rows]),
                  min_eta=if(is.null(f))NA_real_ else min(f$linear.predictors),
                  warnings=paste(unique(result$warnings),collapse=" | "),
                  error=if(is.null(result$error))""else result$error,rng_unchanged=TRUE)
  for(j in seq_along(beta))r[[paste0("beta",j-1L)]]<-beta[j]
  for(i in seq_along(mu))r[[paste0("mu",i-1L)]]<-mu[i]
  r
}
for(c in cases) {
  read_binary<-function(name,n)readBin(c$binary_inputs[[name]],"double",n=n,size=8,endian="little")
  x<-matrix(read_binary("design",c$observations*c$parameters),nrow=c$observations)
  y<-read_binary("response",c$observations);offset<-read_binary("offset",c$observations)
  weights<-read_binary("weights",c$observations);common_start<-read_binary("start",c$parameters)
  runs<-prefixes<-list()
  for(model in c("poisson","nb_fixed","nb_ml")) {
    for(initialization in c("default","shared")) for(maxit in c(100L,1000L)) {
      start<-if(initialization=="shared")common_start else NULL
      r<-fit_one(x,y,offset,weights,start,model,maxit,trace=TRUE)
      runs[[length(runs)+1L]]<-get_row(r,model,initialization,maxit,x,y,offset,weights)
      write_json(r[c("warnings","messages","error","output","rng_unchanged")],
                 file.path(c$directory,paste0("r_trace_",model,"_",initialization,"_",maxit,".json")),
                 auto_unbox=TRUE,pretty=TRUE,digits=17,na="null")
    }
    if(model!="nb_ml") for(maxit in budgets) {
      r<-fit_one(x,y,offset,weights,common_start,model,maxit)
      prefixes[[length(prefixes)+1L]]<-get_row(r,model,"shared",maxit,x,y,offset,weights)
    }
  }
  write.table(do.call(rbind,runs),file.path(c$directory,"r_runs.tsv"),sep="\t",row.names=FALSE,quote=TRUE)
  write.table(do.call(rbind,prefixes),file.path(c$directory,"r_prefix.tsv"),sep="\t",row.names=FALSE,quote=TRUE)
}
write_json(list(R=R.version.string,MASS=as.character(packageVersion("MASS")),
                rng=RNGkind(),seed=seed,fitting_rng_state_unchanged=TRUE,
                reference="Documented glm.fit/glm.nb calls and public trace output; no package source copied"),
           file.path(out,"reference_environment.json"),auto_unbox=TRUE,pretty=TRUE)
