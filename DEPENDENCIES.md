# Dependency and provenance ledger

| Component | Use | Licence | Distribution |
|---|---|---|---|
| RapidGLM code | engine, CLI, harness | MIT | this repository |
| Boost headers (>=1.65) | probability functions, packed DynamicBitset membership, Boost.Test | BSL-1.0 | system dependency, no vendored source |
| LAPACK / LAPACKE | Householder QR, condition estimates and SVD through the C interface; Cholesky factorizations above 16 columns and triangular solves only with `RAPIDGLM_LAPACK_SOLVES` | BSD-3-Clause | linked dependency, no vendored source |
| OpenBLAS (or compatible BLAS/LAPACK) | optimized matrix kernels under LAPACK | BSD-3-Clause for OpenBLAS | linked dependency; thread-safe build with one internal worker for outer OpenMP |
| C++17 standard library (host GCC libstdc++) | language runtime | GPL-3 with GCC Runtime Library Exception 3.1 | dynamically linked; exception permits MIT applications |
| OpenMP (host GCC libgomp; optional) | parallel feature/test loops | GPL-3 with GCC Runtime Library Exception 3.1 | dynamically linked; optional scalar build |
| CMake | build tooling | BSD-3-Clause | build only |
| Python | validation orchestration | PSF-2.0 | validation only |
| NumPy / SciPy | independent numeric checks | BSD-3-Clause | validation only |
| R, MASS, Matrix, SCEPTRE, edgeR, DESeq2, Signac | black-box comparison, data export | separate GPL/LGPL environments; consult each package DESCRIPTION | never linked or shipped |
| glmGamPoi 1.25.1 | quasi-likelihood recipe reference; black-box comparison | MIT (from 1.25; earlier releases GPL-3) | never linked or shipped; no code copied |
| DESeq2 1.48.2 | bulk DE procedure from its paper and documentation; black-box comparison and calibration | LGPL (>= 3) | never linked or shipped; no source read |
| PyDESeq2 0.5.2 | formula for the `legacy_cooks` robust dispersion (`robust_method_of_moments_disp`, `trimmed_cell_variance`); from 1.4.0 the starting dispersion (`fit_rough_dispersions`, `fit_moments_dispersions`) and the floored NB fits (`irls_solver`) of `bulk_de` and `nb_floored_fits`; black-box comparison; RapidGLM inference back end target | MIT, Copyright (c) 2022 Owkin | never linked or shipped; reimplemented in C++ with attribution (notice below) |
| PyDESeq2 0.5.2 (Rust port) | workflow of `DeseqDataSet` and `DeseqStats` ported to Rust in `bindings/rust/src/deseq` | MIT, Copyright (c) 2022 Owkin | ported source in this repository with attribution (notice below) |
| lowess by Alexandre Gramfort (via PyDESeq2) | independent-filtering smoother in `bindings/rust/src/deseq/util.rs` | BSD-3-Clause | ported with attribution |
| jaxQTL 0.3.1 | design of the cis-QTL workflow (`cis_qtl`: score tests against a covariate-only null, permutations that refit the null, the Beta approximation with its chi-square(1, ncp) reference); black-box comparison | MIT, Copyright (c) 2026 Mancuso & Gazal Labs | never linked or shipped; method reimplemented in C++ with attribution (notice below) |
| libm (Rust crate) | erfc and lgamma in the Rust binding | MIT or Apache-2.0 | Cargo dependency |
| Bioconductor airway 1.28.0 | bulk DE validation data | LGPL | validation only; data not committed |

PyDESeq2 provenance: with the author's approval (2026-10-01), the robust
dispersion behind `legacy_cooks` (the default from 1.2.0) follows PyDESeq2's
implementation (`pydeseq2/utils.py`, functions `robust_method_of_moments_disp`,
`trimmed_cell_variance`, `trimmed_variance` and `trimmed_mean`; flagging rule in
`DeseqDataSet.cooks_outlier`). It is reimplemented in C++ as
`detail::legacy_cooks_dispersion` in `include/rapidglm/bulk.hpp`. With the
author's approval (2026-10-17), 1.4.0 also follows PyDESeq2's `irls_solver`
(fitted counts floored at `min_mu` inside the iterations, ridge 1e-6, bounded
coefficients; `detail::floored_nb_fit`) and its starting dispersion, the smaller
of `fit_rough_dispersions` and `fit_moments_dispersions` (the `moment_dispersion`
rule in `bulk_de`). Cite Muzellec
et al. (2023), Bioinformatics 39:btad547. PyDESeq2's licence:

> MIT License
>
> Copyright (c) 2022 Owkin
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Rust port of PyDESeq2: with the author's approval (2026-10-01), the Rust crate's
`deseq` module ports PyDESeq2 0.5.2's workflow from its source (`dds.py`,
`ds.py`, `utils.py`, `preprocessing.py`, the installed release, whose licence is
the MIT notice above), calling RapidGLM for the per-gene fits. PyDESeq2 states
that two of these steps, the rough and moments starting dispersions
(`fit_rough_dispersions`, `fit_moments_dispersions`), follow "the R code", i.e.
DESeq2's; they are ported as PyDESeq2 publishes them. Its lowess is marked
"Authors: Alexandre Gramfort; License: BSD (3-clause); Adapted from
https://gist.github.com/agramfort/850437".

jaxQTL provenance: the cis-QTL workflow (`include/rapidglm/qtl.hpp`) follows the
method of jaxQTL 0.3.1 (Zhang, Kim, Suboc, Mancuso and Gazal 2025), read from its
source, which is MIT-licensed from its first commit: the score test of each
variant against the covariate-only null, permutations of the response and
offset with the null refitted, and the Beta approximation whose reference
non-centrality is solved so that the moment estimate of the first Beta shape is
1. It is reimplemented in C++, not translated. jaxQTL's licence:

> The MIT License (MIT)
>
> Copyright (c) 2026 Mancuso & Gazal Labs
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

edgeR and limma (GPL) were not read: `edger_ql` follows their papers, their
documentation and black-box runs of edgeR 4.6.2 (METHODS.md).

Boost licence: https://www.boost.org/LICENSE_1_0.txt
GCC exception: https://www.gnu.org/licenses/gcc-exception-3.1.html
LAPACK licence: https://github.com/Reference-LAPACK/lapack/blob/master/LICENSE
OpenBLAS licence: https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE
glmGamPoi provenance: the pseudobulk quasi-likelihood test follows the recipe
documented by glmGamPoi (Ahlmann-Eltze and Huber 2020), and the grouped-design
solver takes its idea of solving designs with as many distinct rows as columns
group by group. Only its MIT-licensed
devel commit 95cffb79 (2026-05-26, v1.25.1; licence changed in a9eeed6) was
read, and none of its GPL-era history. Blocks in that commit that declare
another origin were skipped: `src/overdispersion.cpp` lines 37-261 (in-file
"License: LGPL (>= 3)", from DESeq2), `src/beta_estimation.cpp` lines 174-424
(DESeq2 `fitBeta` fork and a one-group solver citing edgeR), `R/overdispersion.R`
lines 259-272 and `R/test_de.R` lines 241-255 (edgeR-derived). RapidGLM's C++
(`include/rapidglm/shrinkage.hpp`, `quasi_likelihood_tests`, the Cox-Reid
adjustment, `detail::RowGroups`) was written independently from the cited
papers and the algebra in METHODS.md; the Cox-Reid
derivatives were derived here. glmGamPoi copyright: Constantin Ahlmann-Eltze
(MIT).

DESeq2 provenance: `include/rapidglm/bulk.hpp` implements the procedure of Love,
Huber and Anders (2014) from the paper, the manual pages and the vignette. No
DESeq2 source was read; PyDESeq2 source was read only for `legacy_cooks` and the
Rust port (above). DESeq2 ran only as a black box
(`scripts/reference_bulk.R`) through its public functions. Details that the
documentation leaves open were calibrated from its outputs, as listed in
[docs/BULK_DE.md](docs/BULK_DE.md#provenance).

No Eigen dependency. No files reused from perturb-resample. Philox implemented
independently from Salmon et al.'s published algorithm, not copied from Random123.
ARS (the AES-based generator of the same paper) likewise: implemented from the
published algorithm, with no Random123 source read. Its test reproduces the
three known-answer vectors for ARS in Random123's `tests/kat_vectors`
(Copyright 2010-2012 D. E. Shaw Research, BSD 3-clause). The exponential
ziggurat follows Marsaglia & Tsang (2000).
Only documentation and data files of GPL comparators may be inspected; their
functions may be executed as black boxes. No reference package implementation
belongs under include/ or src/.

Validation also uses pandas (BSD-3-Clause), h5py (BSD-3-Clause), and externally
installed DESeq2 1.48.2. clang-format (Apache-2.0 with LLVM exception) was used
as development tooling only, installed beneath ignored `validation/local-data/`.
No third-party source was added to this repository.

Black (MIT) was used only to format the Python validation scripts. Rcpp
(GPL-2 or later) is confined to the optional S6 validation bridge; it is not
linked into or required by the installed RapidGLM library or CLI. The S6
simulator and RNG execute from the unchanged external perturb-resample
checkout under the author's explicit reuse instruction; no files from that
checkout are redistributed or copied into the native implementation.

The numerical backend follows FastBMA's use of LAPACK/BLAS, with no FastBMA
application or update source copied. Validation uses LAPACKE 3.10.0 and pthread
OpenBLAS 0.3.20 (`libopenblas0-pthread` / `libopenblas-pthread-dev`, Ubuntu
`0.3.20+ds-1`) with `OPENBLAS_NUM_THREADS=1`, extracted under ignored local data. The
local test binaries link the LAPACKE archive and dynamically link OpenBLAS;
ordinary system installations can use either shared or static LAPACKE.
The earlier serial package failed deterministic concurrent-call tests and is
unsuitable for this outer OpenMP configuration. Serial OpenBLAS requires
`USE_LOCKING=1` for concurrent callers; see the
[upstream build options](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/build_system.md).
