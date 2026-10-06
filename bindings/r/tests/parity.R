# Parity of rapidglm's drop-in fitting with stats::glm and MASS::glm.nb.
library(rapidglm)
set.seed(3)
n <- 500
d <- data.frame(a = rnorm(n), b = factor(sample(letters[1:3], n, TRUE)), depth = rexp(n) + 1)
d$ab <- d$a * 2
d$y_pois <- rpois(n, exp(0.3 + 0.4 * d$a + (d$b == "b") * 0.5) * d$depth)
d$y_bin <- rbinom(n, 1, plogis(0.2 - 0.5 * d$a))
d$succ <- rbinom(n, 10, plogis(0.1 + 0.3 * d$a))
d$fail <- 10 - d$succ
d$y_gauss <- 1 + d$a + rnorm(n)
d$y_nb <- rnbinom(n, mu = exp(1 + 0.5 * d$a), size = 2)
d$w <- runif(n, 0.5, 2)
d$w0 <- ifelse(seq_len(n) %% 7 == 0, 0, d$w)

check <- function(label, new, ref, coef_tol = 1e-8, se_tol = 1e-5) {
  stopifnot(identical(new$method, rapidglm::glm_fit))
  s1 <- coef(summary(new))
  s2 <- coef(summary(ref))
  stopifnot(identical(rownames(s1), rownames(s2)),
            max(abs(s1[, 1] - s2[, 1])) < coef_tol,
            max(abs(s1[, 2] - s2[, 2]) / s2[, 2]) < se_tol,
            isTRUE(all.equal(deviance(new), deviance(ref), tolerance = 1e-10)),
            isTRUE(all.equal(AIC(new), AIC(ref), tolerance = 1e-10)),
            identical(df.residual(new), df.residual(ref)),
            identical(is.na(coef(new)), is.na(coef(ref))))
  cat(label, "ok\n")
}
models <- list(
  poisson_offset = quote(stats::glm(y_pois ~ a + b + offset(log(depth)), poisson, d)),
  poisson_weights = quote(stats::glm(y_pois ~ a + b, poisson, d, weights = w)),
  poisson_zero_weights = quote(stats::glm(y_pois ~ a + b, poisson, d, weights = w0)),
  binomial = quote(stats::glm(y_bin ~ a + b, binomial, d)),
  binomial_factor = quote(stats::glm(factor(y_bin) ~ a, binomial, d)),
  binomial_cbind = quote(stats::glm(cbind(succ, fail) ~ a + b, binomial, d)),
  gaussian = quote(stats::glm(y_gauss ~ a + b, gaussian, d)),
  aliased = quote(stats::glm(y_pois ~ a + ab + b, poisson, d)),
  quasi_fallback = quote(stats::glm(y_pois ~ a, quasipoisson, d)))
for (label in names(models)) {
  ref <- eval(models[[label]])
  new <- update(ref, method = rapidglm::glm_fit)
  check(label, new, ref)
}

ref <- stats::glm(y_pois ~ a + b + offset(log(depth)), poisson, d)
new <- rapidglm::glm(y_pois ~ a + b + offset(log(depth)), poisson, d)
stopifnot(isTRUE(all.equal(anova(ref, test = "Chisq"), anova(new, test = "Chisq"), tolerance = 1e-6)),
          max(abs(predict(ref, d[1:5, ], se.fit = TRUE)$se.fit -
                  predict(new, d[1:5, ], se.fit = TRUE)$se.fit)) < 1e-6,
          max(abs(hatvalues(ref) - hatvalues(new))) < 1e-6,
          max(abs(cooks.distance(ref) - cooks.distance(new))) < 1e-6,
          max(abs(suppressMessages(confint(ref)) - suppressMessages(confint(new)))) < 1e-6,
          isTRUE(all.equal(drop1(ref, test = "Chisq"), drop1(new, test = "Chisq"), tolerance = 1e-6)))
# R's working weights come from the start of its last iteration (differences near
# 1e-6), so influence measures agree to that order, not to rounding.
zr <- stats::glm(y_pois ~ a + b, poisson, d, weights = w0)
zn <- update(zr, method = rapidglm::glm_fit)
stopifnot(length(zn$effects) == length(zr$effects),
          max(abs(hatvalues(zr) - hatvalues(zn))) < 1e-6,
          max(abs(rstandard(zr) - rstandard(zn))) < 1e-6)
cat("glm methods ok\n")

nbr <- MASS::glm.nb(y_nb ~ a + b, d)
nbn <- rapidglm::glm_nb(y_nb ~ a + b, d)
stopifnot(inherits(nbn, "negbin"),
          max(abs(coef(nbr) - coef(nbn))) < 1e-6,
          abs(nbr$theta - nbn$theta) / nbr$theta < 1e-6,
          abs(nbr$SE.theta - nbn$SE.theta) / nbr$SE.theta < 1e-4,
          abs(nbr$twologlik - nbn$twologlik) < 1e-6,
          isTRUE(all.equal(AIC(nbr), AIC(nbn), tolerance = 1e-8)))
nb0r <- MASS::glm.nb(y_nb ~ a, d)
nb0n <- rapidglm::glm_nb(y_nb ~ a, d)
stopifnot(isTRUE(all.equal(anova(nb0r, nbr)[["LR stat."]], anova(nb0n, nbn)[["LR stat."]],
                           tolerance = 1e-6)))
cat("glm_nb ok\n")
