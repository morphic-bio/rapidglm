\name{glm_fit}
\alias{glm_fit}
\alias{glm}
\title{Fit Generalized Linear Models with the RapidGLM Engine}
\description{
  \code{glm_fit} is a fitting method for \code{\link[stats]{glm}} with the
  arguments and value of \code{\link[stats]{glm.fit}}. \code{glm} calls
  \code{stats::glm} with \code{method = rapidglm::glm_fit} as its default.
}
\usage{
glm_fit(x, y, weights = NULL, start = NULL, etastart = NULL, mustart = NULL,
        offset = NULL, family = gaussian(), control = list(), intercept = TRUE,
        singular.ok = TRUE)

glm(formula, family = gaussian, data, weights, subset, na.action, start = NULL,
    etastart, mustart, offset, control = list(...), model = TRUE,
    method = glm_fit, x = FALSE, y = TRUE, singular.ok = TRUE, contrasts = NULL, ...)
}
\arguments{
  \item{x, y, weights, start, etastart, mustart, offset, family, control, intercept, singular.ok}{
    As for \code{\link[stats]{glm.fit}}.}
  \item{formula, data, subset, na.action, model, method, contrasts, ...}{
    As for \code{\link[stats]{glm}}.}
}
\details{
  Poisson (log), binomial (logit) and \code{MASS::negative.binomial} (log)
  families are fitted natively. Gaussian identity fits, other families or
  links, and supplied \code{etastart} or \code{mustart} use
  \code{stats::glm.fit}. Convergence requires the relative deviance change and
  the relative coefficient change to fall below \code{min(1e-9, epsilon)}.
  Columns whose weighted residual norm is below \code{min(1e-7, epsilon/1000)}
  of their norm are aliased (\code{NA}), as in \code{stats::glm}.
  Standard errors use the working weights at the final estimates.
}
\value{A list with the components of \code{\link[stats]{glm.fit}}.}
\examples{
d <- data.frame(x = rnorm(100))
d$y <- rpois(100, exp(0.5 + 0.3 * d$x))
fit <- stats::glm(y ~ x, poisson, d, method = glm_fit)
summary(fit)
}
