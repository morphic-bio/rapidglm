\name{special}
\alias{special}
\alias{special_log}
\alias{special_log1p}
\alias{special_exp}
\alias{special_lgamma}
\alias{special_digamma}
\alias{special_trigamma}
\alias{special_path}
\title{Vectorized Special Functions}
\description{
  The special functions of the RapidGLM engine for double vectors:
  \code{special_log}, \code{special_log1p}, \code{special_exp},
  \code{special_digamma} and \code{special_trigamma} are RapidGLM's (largest
  errors measured 0.67, 0.68, 0.86, 3.7 and 2.7 ulp); they run vectorized (AVX2,
  or Advanced SIMD on AArch64) and give the same bits on every processor, and
  \code{special_path()} names the path taken here. \code{special_lgamma} is the
  C library's \code{lgamma_r}.
}
\usage{
special_log(x, division = NULL)
special_log1p(x, division = NULL)
special_exp(x, division = NULL)
special_lgamma(x, division = NULL)
special_digamma(x, division = NULL)
special_trigamma(x, division = NULL)
special_path()
}
\arguments{
\item{division}{Optional numerical backend policy, \code{"slow"} or \code{"fast"}; \code{NULL} inherits the environment/build default.}
  \item{x}{A numeric vector, matrix or array.}
}
\details{
  As base R's functions, they keep the attributes of \code{x} (dimensions,
  names), return \code{NA} for \code{NA} and \code{NaN} for \code{NaN}, and
  warn when other arguments give \code{NaN}. Unlike \code{\link{digamma}} and
  \code{\link{trigamma}}, \code{special_digamma} and \code{special_trigamma}
  are defined for \code{x >= 0} only: negative arguments give \code{NaN}.
  \code{special_lgamma} is \eqn{\log|\Gamma(x)|}{log|Gamma(x)|} for every
  \code{x}, as \code{\link{lgamma}}.
}
\value{
  A double vector with the attributes of \code{x}; for \code{special_path},
  \code{"avx2"}, \code{"neon"} or \code{"portable"}.
}
\examples{
x <- c(0.5, 1, 2.5, 10, 1e6)
cbind(special_digamma(x, division = NULL), digamma(x))
special_path()
}
