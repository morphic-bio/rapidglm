#pragma once
// Column-major LAPACK routines used by the engine, behind one small interface.
// By default the C interface (LAPACKE) is used, so callers do not depend on a
// Fortran compiler's calling convention. Defining RAPIDGLM_R_LAPACK before
// including RapidGLM instead calls the LAPACK that R links, through R's
// documented Fortran declarations (for R packages).
#ifdef RAPIDGLM_R_LAPACK
#include <R_ext/Lapack.h>
#ifndef FCONE
#define FCONE
#endif
#else
#include <lapacke.h>
#endif

namespace rapidglm::lapack {
#ifdef RAPIDGLM_R_LAPACK
#ifdef La_INT
using integer = La_INT;
#else
using integer = int; // R without La_INT (e.g. 4.3): LAPACK integers are int
#endif
inline integer trcon(char norm, char uplo, char diag, integer n, const double *a, integer lda,
                     double *rcond, double *work, integer *iwork) {
    integer info = 0;
    F77_CALL(dtrcon)(&norm, &uplo, &diag, &n, a, &lda, rcond, work, iwork, &info FCONE FCONE FCONE);
    return info;
}
inline integer potrf(char uplo, integer n, double *a, integer lda) {
    integer info = 0;
    F77_CALL(dpotrf)(&uplo, &n, a, &lda, &info FCONE);
    return info;
}
inline integer potrs(char uplo, integer n, integer nrhs, const double *a, integer lda, double *b,
                     integer ldb) {
    integer info = 0;
    F77_CALL(dpotrs)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info FCONE);
    return info;
}
inline integer potri(char uplo, integer n, double *a, integer lda) {
    integer info = 0;
    F77_CALL(dpotri)(&uplo, &n, a, &lda, &info FCONE);
    return info;
}
inline integer pocon(char uplo, integer n, const double *a, integer lda, double anorm,
                     double *rcond, double *work, integer *iwork) {
    integer info = 0;
    F77_CALL(dpocon)(&uplo, &n, a, &lda, &anorm, rcond, work, iwork, &info FCONE);
    return info;
}
inline integer gesvd(char jobu, char jobvt, integer m, integer n, double *a, integer lda, double *s,
                     double *u, integer ldu, double *vt, integer ldvt, double *work,
                     integer lwork) {
    integer info = 0;
    F77_CALL(dgesvd)(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
                     &info FCONE FCONE);
    return info;
}
inline integer geqrf(integer m, integer n, double *a, integer lda, double *tau, double *work,
                     integer lwork) {
    integer info = 0;
    F77_CALL(dgeqrf)(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}
inline integer ormqr(char side, char trans, integer m, integer n, integer k, const double *a,
                     integer lda, const double *tau, double *c, integer ldc, double *work,
                     integer lwork) {
    integer info = 0;
    F77_CALL(dormqr)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork,
                     &info FCONE FCONE);
    return info;
}
inline integer trtrs(char uplo, char trans, char diag, integer n, integer nrhs, const double *a,
                     integer lda, double *b, integer ldb) {
    integer info = 0;
    F77_CALL(dtrtrs)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info FCONE FCONE FCONE);
    return info;
}
#else
using integer = lapack_int;
inline integer trcon(char norm, char uplo, char diag, integer n, const double *a, integer lda,
                     double *rcond, double *work, integer *iwork) {
    return LAPACKE_dtrcon_work(LAPACK_COL_MAJOR, norm, uplo, diag, n, a, lda, rcond, work, iwork);
}
inline integer potrf(char uplo, integer n, double *a, integer lda) {
    return LAPACKE_dpotrf_work(LAPACK_COL_MAJOR, uplo, n, a, lda);
}
inline integer potrs(char uplo, integer n, integer nrhs, const double *a, integer lda, double *b,
                     integer ldb) {
    return LAPACKE_dpotrs_work(LAPACK_COL_MAJOR, uplo, n, nrhs, a, lda, b, ldb);
}
inline integer potri(char uplo, integer n, double *a, integer lda) {
    return LAPACKE_dpotri_work(LAPACK_COL_MAJOR, uplo, n, a, lda);
}
inline integer pocon(char uplo, integer n, const double *a, integer lda, double anorm,
                     double *rcond, double *work, integer *iwork) {
    return LAPACKE_dpocon_work(LAPACK_COL_MAJOR, uplo, n, a, lda, anorm, rcond, work, iwork);
}
inline integer gesvd(char jobu, char jobvt, integer m, integer n, double *a, integer lda, double *s,
                     double *u, integer ldu, double *vt, integer ldvt, double *work,
                     integer lwork) {
    return LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                               work, lwork);
}
inline integer geqrf(integer m, integer n, double *a, integer lda, double *tau, double *work,
                     integer lwork) {
    return LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, a, lda, tau, work, lwork);
}
inline integer ormqr(char side, char trans, integer m, integer n, integer k, const double *a,
                     integer lda, const double *tau, double *c, integer ldc, double *work,
                     integer lwork) {
    return LAPACKE_dormqr_work(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc, work,
                               lwork);
}
inline integer trtrs(char uplo, char trans, char diag, integer n, integer nrhs, const double *a,
                     integer lda, double *b, integer ldb) {
    return LAPACKE_dtrtrs_work(LAPACK_COL_MAJOR, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}
#endif
} // namespace rapidglm::lapack
