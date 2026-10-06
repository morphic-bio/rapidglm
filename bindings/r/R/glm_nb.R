# Negative binomial GLM with maximum-likelihood theta, returning an object with
# the documented structure of MASS::glm.nb() (class "negbin"; components theta,
# SE.theta, twologlik). Coefficients and theta are estimated jointly by the
# engine; stats::glm() then assembles the model object at that theta.
glm_nb <- function(formula, data, weights, subset, na.action, start = NULL, etastart, mustart,
                   control = glm.control(...), method = "glm.fit", model = TRUE, x = FALSE,
                   y = TRUE, contrasts = NULL, ..., init.theta, link = log) {
  call <- match.call()
  link_name <- deparse(substitute(link))
  if (!identical(link_name, "log") || !missing(etastart) || !missing(mustart)) {
    call[[1L]] <- quote(MASS::glm.nb)
    return(eval(call, parent.frame()))
  }
  # Standard model-frame construction from the matched call.
  frame_call <- call[c(1L, match(c("formula", "data", "subset", "weights", "na.action",
                                   "offset"), names(call), 0L))]
  frame_call$drop.unused.levels <- TRUE
  frame_call[[1L]] <- quote(stats::model.frame)
  mf <- eval(frame_call, parent.frame())
  mt <- attr(mf, "terms")
  Y <- model.response(mf, "numeric")
  X <- if (!stats::is.empty.model(mt)) model.matrix(mt, mf, contrasts) else matrix(, NROW(Y), 0L)
  w <- model.weights(mf)
  if (is.null(w)) w <- rep(1, NROW(Y))
  off <- model.offset(mf)
  if (is.null(off)) off <- rep(0, NROW(Y))
  control <- do.call("glm.control", as.list(control))
  init <- if (missing(init.theta)) 10 else init.theta
  f <- .fit_native(X, Y, w, off, start, "nb", init, TRUE, control, TRUE)
  if (!f$converged) warning("alternation limit reached")
  theta <- f$theta
  beta <- f$coefficients
  beta[f$aliased] <- 0
  fit_call <- call
  fit_call[[1L]] <- quote(stats::glm)
  fit_call$family <- MASS::negative.binomial(theta)
  fit_call$method <- quote(rapidglm::glm_fit)
  fit_call$start <- beta
  fit_call$init.theta <- NULL
  fit_call$link <- NULL
  fit <- eval(fit_call, parent.frame())
  # Observed information for theta at the fitted means (MASS documents SE.theta
  # as an approximation from the observed information).
  mu <- fit$fitted.values
  yy <- fit$y
  pw <- fit$prior.weights
  info <- -sum(pw * (trigamma(theta + yy) - trigamma(theta) + 1 / theta - 2 / (theta + mu) +
                     (yy + theta) / (theta + mu)^2))
  fit$theta <- theta
  fit$SE.theta <- 1 / sqrt(info)
  fit$twologlik <- 2 * f$loglik
  fit$init.theta <- init
  fit$call <- call
  fit$call[[1L]] <- quote(rapidglm::glm_nb)
  fit$aic <- -fit$twologlik + 2 * fit$rank + 2
  fit$method <- method
  class(fit) <- c("negbin", "glm", "lm")
  fit
}
