# Fitting method for stats::glm(): glm(..., method = rapidglm::glm_fit).
# Arguments and returned components follow the documented stats::glm.fit
# interface (?glm, ?family); this file is written from that documentation.

# Gaussian identity fits are a single weighted least-squares solve, which
# glm.fit's own QR already provides along with the qr object glm() stores, so
# they stay with stats::glm.fit.
.native_family <- function(family) {
  name <- family$family
  link <- family$link
  if (identical(name, "poisson") && identical(link, "log")) return("poisson")
  if (identical(name, "binomial") && identical(link, "logit")) return("binomial")
  if (startsWith(name, "Negative Binomial(") && identical(link, "log")) return("nb")
  NULL
}

# theta of a MASS::negative.binomial() family, recovered from its documented
# variance function V(mu) = mu + mu^2 / theta at a scale that avoids cancellation.
.family_theta <- function(family) {
  inverse <- family$variance(1) - 1
  mu <- max(1, 1 / inverse)
  mu^2 / (family$variance(mu) - mu)
}

.fit_native <- function(x, y, weights, offset, start, native, theta, estimate_theta, control,
                        singular.ok) {
  tolerance <- min(1e-9, control$epsilon)
  alias <- min(1e-7, control$epsilon / 1000)
  rapidglm_fit_(x, as.double(y), as.double(weights), as.double(offset),
                if (is.null(start)) double() else as.double(start), native, as.double(theta),
                estimate_theta, tolerance, as.integer(max(control$maxit, 1L)), singular.ok, alias)
}

glm_fit <- function(x, y, weights = NULL, start = NULL, etastart = NULL, mustart = NULL,
                    offset = NULL, family = gaussian(), control = list(), intercept = TRUE,
                    singular.ok = TRUE) {
  control <- do.call("glm.control", control)
  x <- as.matrix(x)
  native <- .native_family(family)
  if (is.null(native) || NCOL(x) == 0L || !is.null(etastart) || !is.null(mustart))
    return(stats::glm.fit(x = x, y = y, weights = weights, start = start, etastart = etastart,
                          mustart = mustart, offset = offset, family = family,
                          control = control, intercept = intercept, singular.ok = singular.ok))
  xnames <- dimnames(x)[[2L]]
  ynames <- if (is.matrix(y)) rownames(y) else names(y)
  nobs <- NROW(y)
  nvars <- ncol(x)
  if (is.null(weights)) weights <- rep.int(1, nobs)
  if (is.null(offset)) offset <- rep.int(0, nobs)
  if (any(weights < 0)) stop("negative weights not allowed")
  # The family's documented initialize expression converts the response (for
  # example binomial factors or two-column counts) and sets n and mustart.
  n <- NULL
  eval(family$initialize)
  if (!is.null(start) && length(start) != nvars)
    stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s",
                  nvars, paste(deparse(xnames), collapse = ", ")), domain = NA)
  theta <- if (native == "nb") .family_theta(family) else 1
  f <- .fit_native(x, y, weights, offset, start, native, theta, FALSE, control, singular.ok)
  if (f$status == "rank_deficient" && !singular.ok) stop("singular fit encountered")
  if (!f$converged)
    warning("glm.fit: algorithm did not converge", call. = FALSE)
  coef <- f$coefficients
  coef[f$aliased] <- NA_real_
  names(coef) <- xnames
  eta <- f$eta
  mu <- f$mu
  if (native == "binomial" && any(mu > 1 - 10 * .Machine$double.eps | mu < 10 * .Machine$double.eps))
    warning("glm.fit: fitted probabilities numerically 0 or 1 occurred", call. = FALSE)
  if (native %in% c("poisson", "nb") && any(mu < 10 * .Machine$double.eps))
    warning("glm.fit: fitted rates numerically 0 occurred", call. = FALSE)
  mu.eta.val <- family$mu.eta(eta)
  residuals <- (y - mu) / mu.eta.val
  good <- weights > 0
  wt <- f$working_weights
  names(residuals) <- names(mu) <- names(eta) <- names(wt) <- ynames
  names(y) <- ynames
  wtdmu <- if (intercept) sum(weights * y) / sum(weights) else family$linkinv(offset)
  nulldev <- sum(family$dev.resids(y, wtdmu, weights))
  dev <- sum(family$dev.resids(y, mu, weights))
  rank <- f$rank
  n.ok <- nobs - sum(weights == 0)
  # glm.fit's qr and effects: base R's QR of sqrt(W) X on positive-weight rows at
  # the final weights, retained columns first and aliased columns after them.
  good <- weights > 0
  kept <- which(!f$aliased)
  dropped <- which(f$aliased)
  sw <- sqrt(wt[good])
  qr <- qr(sw * x[good, kept, drop = FALSE], tol = 0)
  if (length(dropped)) {
    qr$qr <- cbind(qr$qr, qr.qty(qr, sw * x[good, dropped, drop = FALSE]))
    qr$qraux <- c(qr$qraux, rep(0, length(dropped)))
  }
  qr$pivot <- c(kept, dropped)
  qr$rank <- rank
  qr$tol <- min(1e-7, control$epsilon / 1000)
  dimnames(qr$qr) <- list(ynames[good], xnames[qr$pivot])
  Rmat <- qr$qr[seq_len(min(sum(good), nvars)), , drop = FALSE]
  Rmat[row(Rmat) > col(Rmat)] <- 0
  effects <- qr.qty(qr, sw * ((eta - offset)[good] + residuals[good]))
  names(effects) <- c(xnames[qr$pivot[seq_len(rank)]], rep.int("", sum(good) - rank))
  aic.model <- family$aic(y, n, mu, weights, dev) + 2 * rank
  list(coefficients = coef, residuals = residuals, fitted.values = mu, effects = effects,
       R = Rmat, rank = rank, qr = qr, family = family, linear.predictors = eta,
       deviance = dev, aic = aic.model, null.deviance = nulldev, iter = f$iterations,
       weights = wt, prior.weights = weights, df.residual = n.ok - rank,
       df.null = n.ok - as.integer(intercept), y = y, converged = f$converged,
       boundary = f$boundary)
}

# stats::glm() with the RapidGLM fitting method as its default.
glm <- function(formula, family = gaussian, data, weights, subset, na.action, start = NULL,
                etastart, mustart, offset, control = list(...), model = TRUE,
                method = glm_fit, x = FALSE, y = TRUE, singular.ok = TRUE, contrasts = NULL,
                ...) {
  call <- match.call()
  call[[1L]] <- quote(stats::glm)
  if (missing(method)) call$method <- quote(rapidglm::glm_fit)
  fit <- eval(call, parent.frame())
  fit$call[[1L]] <- quote(rapidglm::glm)
  fit
}
