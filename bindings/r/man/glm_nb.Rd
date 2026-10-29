\name{glm_nb}
\alias{glm_nb}
\title{Negative Binomial Generalized Linear Models}
\description{
  Fits a negative binomial GLM with log link, estimating the coefficients and
  the size parameter \code{theta} jointly by maximum likelihood, and returns an
  object of class \code{"negbin"} as \code{\link[MASS]{glm.nb}} does.
}
\usage{
glm_nb(formula, data, weights, subset, na.action, start = NULL, etastart, mustart,
       control = glm.control(...), method = "glm.fit", model = TRUE, x = FALSE,
       y = TRUE, contrasts = NULL, ..., init.theta, link = log, division = NULL)
}
\arguments{
\item{division}{Optional numerical backend policy, \code{"slow"} or \code{"fast"}; \code{NULL} inherits the environment/build default.}
  \item{formula, data, weights, subset, na.action, start, etastart, mustart, control,
        method, model, x, y, contrasts, ..., init.theta, link}{
    As for \code{\link[MASS]{glm.nb}}. Links other than log, or supplied
    \code{etastart} or \code{mustart}, are passed to \code{MASS::glm.nb}.}
}
\value{
  An object of class \code{c("negbin", "glm", "lm")} with components
  \code{theta}, \code{SE.theta} (from the observed information at the fitted
  means) and \code{twologlik}, in addition to those of a \code{glm} fit.
}
\examples{
d <- data.frame(x = rnorm(200))
d$y <- rnbinom(200, mu = exp(1 + 0.5 * d$x), size = 2)
fit <- glm_nb(y ~ x, d)
summary(fit)
}
