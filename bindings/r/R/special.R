# Special functions of double vectors through the C interface (man/special.Rd):
# RapidGLM's log, log1p, exp, digamma and trigamma and the C library's lgamma. As
# base R's functions they keep the argument's attributes, return NA for NA and
# NaN for NaN, and warn when other arguments give NaN (src/special.cpp).
special_log <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "log")
}
special_log1p <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "log1p")
}
special_exp <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "exp")
}
special_lgamma <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "lgamma")
}
special_digamma <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "digamma")
}
special_trigamma <- function(x, division = NULL) {
  previous <- .division_set(division)
  on.exit(.division_restore(previous), add = TRUE)
  rapidglm_special_(x, "trigamma")
}
special_path <- function() rapidglm_special_path_()


.division_set <- function(division) {
  if (is.null(division)) return(NULL)
  if (length(division) != 1L || is.na(division) || !division %in% c("slow", "fast"))
    stop("division must be slow or fast", call. = FALSE)
  rapidglm_division_set_(if (division == "fast") 1L else 0L)
}

.division_restore <- function(previous) {
  if (!is.null(previous)) rapidglm_division_set_(previous)
  invisible(NULL)
}
