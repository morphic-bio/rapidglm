# edgeR 4.6.2 quasi-likelihood workflow: clean-room specification

Date: 2026-10-16. Versions probed: edgeR 4.6.2, limma 3.64.1, R 4.5.1 (reference BLAS).

## Provenance and labels

This specification was written from published papers, the rendered help pages,
the edgeR User's Guide (revised 16 April 2024), the edgeR and limma NEWS files
and black-box runs of the functions. No edgeR or limma source code was read,
printed or paraphrased, and none of the GPL ports of limma were consulted.

Each statement carries a label:

- **[D: source]** comes from a document: a paper section, a help page, the
  User's Guide (UG) or NEWS.
- **[P: probe]** was established by running edgeR or limma as a black box and
  inspecting outputs. Probe scripts are in `probes/` and their outputs in
  `probes/out/`. The appendix lists them.
- **[O]** marks a point that is still open. Each comes with a proposal for
  settling it.

Notation: G genes (after filtering), n samples, p design columns, d = n - p,
y_gi counts, X the design, j the tested column(s), q the number of tested
columns. log is the natural log.

Main sources:

- Robinson & Oshlack 2010, Genome Biol 11:R25 (TMM).
- McCarthy, Chen & Smyth 2012, NAR 40:4288 (NB GLM, Cox-Reid adjusted profile
  likelihood (APL)).
- Chen, Lun & Smyth 2014, the edgeR chapter preprint, sections 3.3-3.4
  (weighted-likelihood empirical Bayes, prior df).
- Chen, Lun & Smyth 2016, F1000Research 5:1438 (QL pipeline, filtering).
- Lund et al. 2012, SAGMB 11(5) (QL F-test).
- Smyth 2004, SAGMB 3(1) (squeezeVar).
- Phipson et al. 2016, AOAS 10:946.
- Lun & Smyth 2017, SAGMB 16:83.
- Chen, Chen, Lun, Baldoni & Smyth 2025, NAR 53(2):gkaf018 ("edgeR v4").
- Baldoni et al. 2024, NAR Genom Bioinform 6:lqae151.
- Help pages: glmQLFit, glmQLFTest, filterByExpr, normLibSizes, aveLogCPM,
  addPriorCount, glmFit, mglmLevenberg, estimateGLMCommonDisp, dispCoxReid,
  adjustedProfileLik and estimateDisp from edgeR; squeezeVar, fitFDist,
  fitFDistUnequalDF1, chooseLowessSpan and weightedLowess from limma.

## 0. The workflow

[D: UG 1.4 "Quick start", UG 2.7, 2.12; glmQLFit help; NAR v4 "In edgeR v4, one
can proceed directly from normalization to glmQLFit"]

```
y    <- DGEList(counts)                       # lib.size = column sums
keep <- filterByExpr(y, design)               # UG 2.7
y    <- y[keep, , keep.lib.sizes = FALSE]     # library sizes recomputed
y    <- normLibSizes(y)                       # TMM
fit  <- glmQLFit(y, design)                   # legacy = FALSE, dispersion = NULL
res  <- glmQLFTest(fit, coef = j)
tab  <- topTags(res, n = Inf)                 # BH FDR
```

- estimateDisp is optional in edgeR 4 [D: UG 2.12.1]. If it was run, glmQLFit
  takes the NB dispersion from it instead (section 6.2b), so the results change.
- legacy = FALSE has been the default since edgeR 4.2.0 [D: NEWS 4.2.0].
- For paired or blocked designs, the UG recommends filtering on the factor of
  interest (`filterByExpr(y, group = Treatment)`) rather than on the full
  design [D: UG 2.7]. The reference runs here use the full design throughout.

## 1. Library sizes and offsets

- L_i = Σ_g y_gi over all genes, for the filter.
- After `y[keep, , keep.lib.sizes = FALSE]`, L_i is recomputed over the kept
  genes [D: UG 2.7].
- The effective library size is N_i = L_i · f_i, where f_i is the TMM factor.
- The GLM offset is o_i = log N_i [D: normLibSizes help, glmFit help
  (`offset = getOffset(y)`)].

## 2. filterByExpr

Defaults are min.count = 10, min.total.count = 15, large.n = 10 and
min.prop = 0.7 [D: help].

The library sizes used are L for a matrix. For a DGEList they are the
effective sizes L · f [P1: reproduced only with L · f once factors are set].
In the workflow order the factors are still 1 at this point.

The rule [D: help; P1 exact on 44 random cases; P22 tolerance]:

1. h_i = diag(X (XᵀX)⁻¹ Xᵀ) and m = 1 / max_i h_i ("minimum inverse leverage").
   When `group` is given, m is the smallest group size.
2. If m > large.n, then m ← large.n + (m − large.n) · min.prop.
3. CPM_gi = y_gi · 10⁶ / L_i and cutoff = min.count / median(L) · 10⁶.
4. Keep gene g if #{i : CPM_gi ≥ cutoff} ≥ m − 1e-14 and Σ_i y_gi ≥ min.total.count.

Two details matter for reproducing ties:

- The arithmetic order. P1c found that only y·10⁶/L or y/(L·10⁻⁶) for the CPM,
  together with min.count/median·10⁶ for the cutoff, reproduce edgeR for
  counts that sit exactly on the cutoff.
- The tolerance on m lies between 1e-14 and 1e-13 [P22]. A design with
  1/max h = 3 + 1e-13 drops a gene that has exactly 3 samples above the cutoff,
  and 3 + 1e-14 keeps it.

## 3. TMM normalization (normLibSizes, method = "TMM")

[D: Robinson & Oshlack 2010, normLibSizes help; P1b reproduces 88 cases
(random, many ties and zeros) to 5e-15]

1. Drop genes whose counts are zero in every sample (they do not affect the
   factors [D]). L_k = column sums of the matrix given; in the workflow, the
   filtered counts.
2. **Reference column r**: the column whose upper quartile of y_gk/L_k is
   closest to the mean of the upper quartiles [D]. The quartile is R's type-7
   quantile at p = 0.75 [P1]. The tie rule was not probed; this spec assumes
   the first such column.
3. For each column k, use the genes with y_gk > 0 and y_gr > 0:
   - M_g = log2((y_gk/L_k) / (y_gr/L_r))
   - A_g = (log2(y_gk/L_k) + log2(y_gr/L_r)) / 2
   - v_g = (L_k − y_gk)/(L_k y_gk) + (L_r − y_gr)/(L_r y_gr), the delta-method
     variance [D].

   Compute M and A exactly as written. Ranks of tied values decide the trim,
   and other algebraically equal forms change floating-point ties: with
   A = ½ log2(product), 10 of 88 cases change [P1b].
4. With n the number of such genes and average ranks for ties, keep the genes
   that satisfy both:
   - floor(0.3 n) + 1 ≤ rank(M) ≤ n − floor(0.3 n)
   - floor(0.05 n) + 1 ≤ rank(A) ≤ n − floor(0.05 n)

   Defaults: logratioTrim = 0.3, sumTrim = 0.05 [D]; trim bounds [P1].
   Acutoff = −1e10 removes nothing in practice.
5. log2 f_k = Σ_keep (M_g / v_g) / Σ_keep (1 / v_g), so f_r = 1.
6. f ← f / exp(mean(log f)), so the factors multiply to 1 [D].

[O] Degenerate cases were not probed: a pair with no usable genes, or an
all-zero sample. Proposal: probe with a sample of all zeros and with two
disjoint supports.

## 4. AveLogCPM

[D: aveLogCPM, addPriorCount and mglmOneGroup help; P1 to 1e-11, P9 exact
`fit$AveLogCPM == aveLogCPM(y)`]

1. Prior count c = 2. The dispersion is 0.05, aveLogCPM's default; the DGEList
   route computes AveLogCPM before any NB dispersion exists [P9].
2. p_i = c · N_i / mean(N), y*_gi = y_gi + p_i and o*_i = log(N_i + 2 p_i)
   [P1: addPriorCount black box].
3. β_g solves Σ_i (y*_gi − μ_gi)/(1 + 0.05 μ_gi) = 0 with μ_gi = exp(β_g + o*_i).
   This is the one-group NB MLE. Fisher scoring from log(Σ y*/Σ exp o*)
   converges in a few steps.
4. AveLogCPM_g = (β_g + log 10⁶) / log 2.

The matrix (default) method of glmQLFit with a user-supplied dispersion φ
computes its AveLogCPM with dispersion φ instead. P16n reproduces s2.prior
exactly only with that covariate. The DGEList workflow never takes this path.

## 5. NB GLM, unit deviance and coefficient shrinkage

[D: glmFit, mglmLevenberg help; McCarthy et al. 2012]

- log μ_gi = x_iᵀ β_g + o_i, with Var = μ + φ μ².
- edgeR fits by Fisher scoring with Levenberg damping (maxit 200, tol 1e-6),
  or by group-wise MLEs for designs that reduce to a one-way layout
  (designAsFactor). Any method that converges to the MLE will do.
- Unit deviance:
  d(y, μ, φ) = 2 [ y log(y/μ) − (y + 1/φ) log((1 + φ y)/(1 + φ μ)) ],
  with the Poisson limit at φ = 0 [D: nbinomUnitDeviance help, standard].
  The deviance is the sum over samples.
- **Shrunken coefficients**, used for the reported logFC only, are the MLE of
  the same model fitted to y_gi + p_i with offsets log(N_i + 2 p_i), where
  p_i = 0.125 · N_i / mean(N) (prior.count = 0.125).
  - Source: [D: glmFit help (prior.count = 0.125, "Phipson 2013"),
    addPriorCount help]
  - Agreement: [P9, P26: to 2e-15 at the working dispersion φ_w of 6.4]
- Deviances, fitted values, leverages and tests all use the unshrunk fit at
  the same dispersion [P9]. `fit$unshrunk.coefficients` holds that fit.

## 6. glmQLFit(y, design) with legacy = FALSE

### 6.1 Inputs

The inputs are the counts, the offsets o_i, AveLogCPM and the design X.
Weights default to 1. abundance.trend = TRUE and robust = FALSE.

### 6.2 NB dispersion φ (when the DGEList has no trended dispersions)

[D: glmQLFit help "common dispersion of the top `top.proportion` of genes by
AveLogCPM"; NAR v4]

1. **Top proportion.**
   t = chooseLowessSpan(G·√d, small.n = 20, min.span = 0.02, power = 1/3),
   which equals 0.02 + 0.98 · (20 / (G √d))^(1/3), and 1 when G√d is at most 20.
   - Source: [D: glmQLFit help, chooseLowessSpan help]
   - Agreement: [P2 exact on 6 datasets, e.g. airway t = 0.106383343260366]
2. **Top genes.** T is the ceiling(t · G) genes with the largest AveLogCPM.
   - Agreement: [P2: ceiling matches all 6 datasets; round and floor fail on
     kang and sim3b]
   - Tie order among equal AveLogCPM: [O], immaterial in practice.
3. **Common Cox-Reid dispersion** of T:
   φ = argmax Σ_{g∈T} APL_g(φ), where
   APL_g(φ) = ℓ_g(β̂_g(φ); φ) − ½ log det(Xᵀ W_g X), with W_gi = μ_gi/(1 + φ μ_gi).
   - ℓ is the full NB log-likelihood at the unshrunk MLE β̂_g(φ) with offsets
     o_i; the constants do not matter.
   - Source: [D: McCarthy 2012, adjustedProfileLik help]
   - Agreement: [P24: equals adjustedProfileLik to 2e-10]
4. **Optimizer.** R's `optimize` (Brent 1973, `fmin`) maximizes over
   s = φ^(1/4) on the interval [0, 4^(1/4)] with tol = 1e-5, and φ = s⁴.
   - Source: [D: dispCoxReid help gives the interval c(0, 4) and
     tol = 1e-5]
   - Agreement: [P23: the quarter-root scale reproduces edgeR to every printed
     digit on 4 datasets. Linear, square-root and log scales do not. The exact
     argmax differs from edgeR's by up to 6e-6 relative.]
   - To match bit for bit, implement Brent's fmin. Brent's algorithm is
     published, and Netlib's `fmin` is public domain.
5. **Row limits.** estimateGLMCommonDisp uses min.row.sum = 5 and
   subset = 10000: above 10000 rows, systematicSubset picks rows evenly spaced
   by AveLogCPM [D: estimateGLMCommonDisp, dispCoxReid help].
   - [O] Not exercised here: |T| ≤ 2069 on our data. It needs G above about
     170k at d = 3. Proposal: probe with a synthetic G = 250k panel.
6. `fit$dispersion` reports φ, not the working dispersion φ_w of section 6.4.

### 6.2b NB dispersion when the DGEList carries estimateDisp output

φ is the mean of `trended.dispersion` over the ceiling(0.1 · G) genes with the
largest AveLogCPM. In this case `top.proportion` is not returned.

- Source: [D: NEWS 4.4.0 "the dispersion from the mean of the right tail
  values of the trended dispersions"]
- Agreement: [P21 exact, relative error 0 to 2e-16, on airway, kang and sim8]

The rest proceeds as in 6.3 to 6.6.

### 6.3 Adjusted unit deviances and degrees of freedom at working scale a

[D: NAR v4, "Correcting quasi-likelihood for bias": unit deviances are
approximated by scaled chi-square variables on fractional df, matching the
mean and variance exactly, given the fitted mean and the NB dispersion;
glmQLFit help (`leverage`, `unit.deviance.adj`, `unit.df.adj`,
`df.residual.adj`)]

The exact form below is [P3, P4-P6, P6g, P8], checked against
`keep.unit.mat = TRUE` output. For a scale a ≥ 1 (section 6.4):

1. Fit the unshrunk NB GLM at φ_w = φ/a to get μ_gi [P9, P16l: edgeR's final
   coefficients equal `glmFit(..., dispersion = φ/a)` exactly].
2. h_gi is the diagonal of W^{1/2} X (Xᵀ W X)⁻¹ Xᵀ W^{1/2}, with
   W_gi = μ_gi/(1 + φ_w μ_gi) [P3: matches the returned leverage].
3. Form divided counts and means: z = y/a and m = μ/a.
   - Let U = d(z, m, φ) with z ~ NB(mean m, dispersion φ).
   - E(m, φ) and V(m, φ) are the mean and variance of U.
   - k = 2E²/V and s = V/(2E), so that U ≈ s · χ²_k.
4. Per observation:
   - unit.deviance.adj_gi = a · d(z_gi, m_gi, φ) / s(m_gi, φ)
     = d(y_gi, μ_gi, φ/a) / s(m_gi, φ)
   - unit.df.adj_gi = (1 − h_gi) · k(m_gi, φ)
5. Per gene:
   - deviance.adj_g = Σ_i unit.deviance.adj_gi
   - df.residual.adj_g = Σ_i unit.df.adj_gi
   - s2_g = deviance.adj_g / df.residual.adj_g [P3: sums exact]

Consequences:

- k → 0 as m → 0, so groups of zeros lose their df automatically, which
  generalizes Lun & Smyth 2017.
- df.residual.adj is fractional and can exceed d, because k > 1 for small m
  when φ is large.
- At a = 1 this is the plain NB moment-matched deviance.

**Moments.** edgeR's internal E and V are not exact sums [P8: implied moments
E_e = K/R and V_e = 2K/R² against exact sums over the NB pmf; table
`probes/out/p8_moments.tsv`].

- For φ ≤ 0.3, s agrees with the exact value to within 3e-5 for m < 1 and
  1.2e-3 for m ≥ 1, and k to within 4e-4.
- For φ = 1 they agree to within 3e-3.
- For φ = 3 and m ≥ 100 they differ by up to 12%. For φ ≥ 1, edgeR's factors
  become constant above m ≈ 3e3 to 7e3; for φ ≤ 0.1 they keep changing.
- Exact moments reproduce edgeR's s2 to a median of 2e-4 and a maximum of
  1.5e-3 on the reference datasets [P25].
- [O] edgeR's own approximation scheme is unknown. Proposal: probe a dense
  (m, φ) grid with anchor genes, as in P8, to map the switch points and the
  large-m behaviour, then fit them. Exact moments are adequate for
  typical bulk data (φ < 0.5).

### 6.4 average.ql.dispersion a (two passes)

[D: glmQLFit help "average quasi-dispersion, used to scale the NB dispersion
after estimating the quasi-dispersions". The rule, the direction (φ/a) and
the number of passes are all from P11-P16.]

```
rule(s2) = max(1, Q_0.9( f_g^4 ))
  f = lowess(x = AveLogCPM, y = s2^(1/4), f = 0.5, iter = 3), evaluated at every gene
  Q_0.9 is the type-7 0.9 quantile over the G genes
a1 = rule(s2 from 6.3 at a = 1)
a2 = rule(s2 from 6.3 at a = a1)     # includes the refit at φ/a1
a  = a2;  final quantities = 6.3 at a2 (refit at φ/a2)
```

**The lowess.** This is R's `lowess`: Cleveland's LOWESS with 3 robustness
iterations. Its default delta is 0.01 · range(x) [D: R lowess documentation].
Values at ties are averaged.

**Evidence for the rule** [P12-P14d]:

- On designed panels with a constant covariate it reproduces edgeR to 1e-7.
  Those panels pin the quarter-root scale, the bisquare weights at 6 × median
  absolute residual and the 3 iterations.
- With the covariate it reproduces edgeR to 1.2e-6 on 4 designed and 4 real
  datasets (span 0.5, 0.9 quantile).

**Evidence for two passes** [P16b, P16h, P16k]:

- Where the fits do not depend on the dispersion (Poisson working model, or
  intercept-only with equal libraries), a2 reproduces edgeR to ≤ 1e-4 and a1
  or a3 do not.
- On sim8 and thyroid32 a2 is within 1e-4 of edgeR, with
  sim8 = 1.589183 against 1.589321.

**Open discrepancies.** [O] Two cases remain unreproduced:

- **Boundary genes.** These are genes whose fit reaches the boundary, such as
  a single nonzero count under a covariate design, in sparse unfiltered data.
  There the two-pass emulation deviates strongly (P16i, P16p: 1.69 against
  2.93). edgeR reports deviance.adj = 0 and df.residual.adj ≈ 1e-13 for these
  genes. It evidently drives such fits closer to the boundary than glmFit
  does, and how they enter the rule was not settled.
- **Unequal libraries with large dispersion.** With unequal library sizes and
  φ ≥ 0.4 there are smaller deviations of 0.1-2% (P16j, P16p).

After filterByExpr the emulation agrees to 0.1% (P16p "filtered"), and the
reference datasets are unaffected. Proposals:

- Repeat P16p with gene subsets to see which genes move a.
- Test the rule on s2 with the boundary genes excluded by df.residual.adj
  below a threshold, together with edgeR's own boundary fits.
- Probe the moment approximations as in 6.3.

Interpretation: the model's variance becomes a·μ + φ·μ² = a(μ + (φ/a)μ²). This
is a Poisson-level inflation, the same structure as the "divided counts" of
Baldoni et al.

### 6.5 Empirical Bayes moderation

[D: glmQLFit help ("calls the limma function squeezeVar"); UG 2.12.2 ("limma's
empirical Bayes hyperparameter estimation was revised in limma 3.62.0 ...
fitFDistUnequalDF1 ... edgeR v4.4.0 onwards"); squeezeVar help (legacy = NULL
uses fitFDistUnequalDF1 whenever df are unequal); fitFDistUnequalDF1 help
("downweighting of observations with lower degrees of freedom ... weighted
moment estimation for scale and profile maximum likelihood for df2")]

[P9]: `squeezeVar(s2, df.residual.adj, covariate = AveLogCPM)` reproduces
`s2.prior`, `df.prior` and `s2.post` to 1e-10.

The inputs are x_g = s2_g, d_g = df.residual.adj_g and c_g = AveLogCPM_g.

1. **Floor.** x_g ← max(x_g, 1e-12 · median(x)).
   - Agreement: [P20c exact; floors of 1e-5 to 1e-10 times the median, and
     absolute floors, fail]
   - Source: [D: limma's warning "Zero sample variances detected, have been
     offset away from zero"]
2. **Log-moment terms.**
   - e_g = log x_g − ψ(d_g/2) + log(d_g/2), where ψ is the digamma function;
     E[log(χ²_d/d)] = ψ(d/2) − log(d/2) [D: Smyth 2004].
   - Weights w_g = 1/ψ′(d_g/2), the inverse variance of log(χ²_d/d)
     [D: Smyth 2004].
   - Agreement: [P18: without a covariate this weighting reproduces the
     scale to about 1e-5; equal, d-proportional and other weightings fail.
     P19e: exact with a covariate]
3. **Trend.**
   t_g = weightedLowess(x = c, y = e, weights = w, span = σ(G), iterations = 1)
   from limma. Its algorithm is documented on its help page: tricube LOWESS
   in which the prior weights define the window by total weight and multiply
   the local weights; no robustness iterations.
   - σ(G) = 1 for G ≤ 500, otherwise 0.3 + 0.7 · (500/G)^(1/3). This equals
     chooseLowessSpan(G/10) with its defaults small.n = 50, min.span = 0.3,
     power = 1/3.
   - Agreement: [P19, P19d: the perturbation pattern shows a local smoother,
     and the span is exact for G = 30 to 20000; P19e: exact to 7e-14 with
     unequal d]
   - Without a covariate, t = Σ w e / Σ w.
   - [O] weightedLowess's default anchor spacing (npts = 200, delta = NULL)
     is not fully documented ("roughly range/npts"). It is 0 when the number
     of distinct x is at most 200. For larger G it is data-dependent and is
     not range/200, range/199 or 0.01 · range [P27]. Using delta = 0 changes
     the trend by ≤ 1e-3 on the log scale. Proposal: work out the anchor rule
     by matching anchor counts on designed x grids.
4. **Scale as a function of the prior df d0.**
   log s0²_g(d0) = t_g + ψ(d0/2) − log(d0/2).
   - Agreement: [P18, P19e, P20: exact at limma's d0. A gene whose s2 sits
     exactly on the trend gets s0² = 3.564 · s2 at d = 1, because
     exp(−ψ(½) + log ½) = 3.564 (P12).]
5. **Prior df.**
   d0 = argmax Σ_g [ log f_F(x_g / s0²_g(d0); d_g, d0) − log s0²_g(d0) ],
   the profile likelihood of the scaled F model [D: "profile maximum
   likelihood for df2"].
   - Agreement: [P20: the exact argmax is within 2e-4 relative of limma's
     d0. limma's one-dimensional optimizer is inexact.]
   - When the likelihood keeps rising, edgeR returns df.prior = 8134.84478
     [P6, P7, P10, P11: the same value in every such case].
   - [O] The optimizer is unknown. Proposal: return the argmax, or cap
     anything above 8134.84478 at that value.
6. **Posterior.**
   - s2.prior_g = s0²_g(d0)
   - s2.post_g = (d0 · s2.prior_g + d_g · s2_g)/(d0 + d_g), with the unfloored s2
   - Source: [D: Smyth 2004]; agreement: [P9 exact]
   - df.prior is a single number because robust = FALSE.

If every df.residual.adj were exactly equal, squeezeVar's default would fall
back to limma's legacy fitFDist [D: squeezeVar help]. Fractional df make this
practically impossible.

## 7. glmQLFTest(fit, coef = j)

[D: help; P9 exact on 4 datasets, including 2 with a > 1]

- The null fit is the unshrunk NB GLM without the column(s) j, at φ_w = φ/a.
- LR_g = deviance_null − deviance_full. These are ordinary, unadjusted
  deviances at φ/a.
- F_g = LR_g / (q · s2.post_g).
- df.total_g = df.prior + df.residual.adj_g [D: help, which says
  "df.residual.zeros" in its legacy wording; P9]. No cap was seen in those
  larger calibration panels. A later public black-box probe of the 12-gene,
  8-sample CLI fixture reports `df.total=72`; RapidGLM's uncapped expression
  gives about 8140.82. This small-panel difference is retained and documented
  in [EDGER_SMALL_PANEL_LIMITATION.json](../validation/receipts/v1.4.0-r2/EDGER_SMALL_PANEL_LIMITATION.json).
  One probe does not establish the general reference cap rule.
- PValue_g = P(F(q, df.total_g) > F_g) [P9; P on outputs to 1e-11].
- poisson.bound applies only when legacy = TRUE [D].
- Output columns:
  - logFC = shrunk coefficient j / log 2 [P].
  - logCPM = AveLogCPM [P].
  - FDR = Benjamini-Hochberg over the tested genes (topTags adjust.method = "BH")
    [D: topTags help; P9].

## 8. Legacy QL (legacy = TRUE), for orientation

[D: glmQLFit help, UG 2.12.1, Lun & Smyth 2017]

- It requires trended NB dispersions, from estimateDisp.
- Its quasi-dispersion is deviance / df.residual.zeros, with zero groups
  removed from the df.
- It uses limma's legacy fitFDist and the Poisson bound in glmQLFTest.

It is not the default and is not specified further here.

## 9. The classic LRT path (second reference output)

**estimateDisp(y, design)** [D: help; chapter preprint 3.3-3.4; McCarthy 2012]:

1. APL_g(φ) is evaluated on a grid of grid.length = 21 points over
   grid.range = c(−10, 10) on the log2 scale around the trend.
2. The common dispersion maximizes the mean APL.
3. The locally shared APL is a tricube-weighted (loess degree 0) average over
   genes near in AveLogCPM; the default trend.method is "locfit".
4. The trended dispersion maximizes the locally shared APL.
5. The tagwise dispersion maximizes APL_g + G0 · APLS_g, with G0 = d0 / d_g.
6. d0 comes from limma's F-distribution fit to the mean deviances.
7. min.row.sum = 5 and tol = 1e-6.

Probe results:

- span = 0.25 + 0.75 · √(50/G) = chooseLowessSpan(G, small.n = 50,
  min.span = 0.25, power = 1/2) [P: exact on all 8 datasets].
- prior.n = prior.df / (n − p) [P: exact].

**glmFit and glmLRT** [D: help]:

- glmFit uses the tagwise dispersions.
- LR = deviance_null − deviance_full and PValue = P(χ²_q > LR)
  [P: outputs to 1e-12; R returns denormals where SciPy underflows to 0].

[O] The grid interpolation (maximizeInterpolant), the locfit trend and the
prior-df step were not probed. They are not needed for the QL workflow.
Proposal: probe them only if RapidGLM implements the LRT.

## 10. Differences from RapidGLM's current `test=ql`

RapidGLM's test follows Lund et al. 2012 with glmGamPoi's recipe (METHODS.md
"quasi-likelihood test", docs/QL_TEST.md).

| Step | RapidGLM `test=ql` | edgeR 4.6.2 default |
|---|---|---|
| Filtering | none in `pseudobulk-de` | filterByExpr (section 2), then library sizes recomputed |
| Normalization | caller's offsets or size factors | TMM effective library sizes (3) |
| NB dispersion used in the fits | Cox-Reid φ_g per gene; trend φ_t = weighted running median against mean fitted count; fits at θ = 1/φ_t | one constant: the Cox-Reid common dispersion of the top ceiling(t·G) genes by AveLogCPM (6.2), divided by a from the two-pass lowess rule (6.4) |
| Quasi-dispersion | s2 = (1 + m φ_g)/(1 + m φ_t) on n − rank df | adjusted deviance / adjusted df from moment-matched scaled chi-square unit deviances with leverage (6.3); fractional df, zero groups lose df |
| Abundance covariate | log mean fitted count | AveLogCPM: log2 average CPM, prior count 2, dispersion 0.05 (4) |
| Prior trend | natural cubic spline in log m (knots at range and thirds), ML for s0² | weighted LOWESS of log-moment residuals, weights 1/ψ′(d/2), span σ(G), one iteration (6.5) |
| Prior df | ML profile on a log grid [1e-3, 1e7], with s0² profiled by Newton | profile ML with s0²(d0) tied to the trend by ψ(d0/2) − log(d0/2); x floored at 1e-12·median |
| LR numerator | deviance difference at φ_t (per gene) | deviance difference at the single φ/a |
| Denominator df | d0 + (n − rank) | d0 + df.residual.adj (fractional) |
| logFC | MLE | MLE of the counts plus prior count 0.125 (shrunk) |

RapidGLM already has the pieces that carry over: Cox-Reid APL, Brent-type 1-D
optimization, NB unit deviance, leverage and the scaled-F likelihood.

The new pieces are:

- filterByExpr
- TMM with the exact tie arithmetic
- AveLogCPM
- exact NB unit-deviance moments
- the divided-count adjustment and the two-pass lowess rule
- weighted LOWESS
- prior-count logFC

## 11. Open points

| # | Point | Effect on our data | Proposal |
|---|---|---|---|
| O1 | edgeR's unit-deviance moment approximation (6.3) | s2 within 1.5e-3; p within 0.5% in log10 | dense (m, φ) probe as in P8; or accept exact moments |
| O2 | average.ql.dispersion with many boundary fits; 0.1-2% gaps at large φ with unequal libraries (6.4) | none after filtering (reference sets reproduced to ≤ 2.7e-4) | gene-subset probes; boundary-gene treatment |
| O3 | limma's df2 optimizer and the 8134.84478 cap (6.5) | d0 within 2e-4 | accept, or reproduce the cap |
| O4 | weightedLowess default delta for more than 200 distinct x (6.5) | trend within 1e-3 | match anchor counts on designed x grids |
| O5 | the subset = 10000 rule for the common dispersion when ceiling(t·G) > 10000 (6.2) | not reached | G = 250k synthetic probe |
| O6 | TMM degenerate cases (3) | none | zero-sample and disjoint-support probes |
| O7 | estimateDisp internals (9) | LRT path only | probe only if needed |

## 12. Evidence that the specification reproduces edgeR

`probes/emulate_full.R` implements sections 2 to 7 from this document. It uses
edgeR and limma only as black boxes for GLM fits at a fixed dispersion (glmFit),
for the Cox-Reid APL (adjustedProfileLik) and for limma's documented
weightedLowess. Moments are exact sums, and d0 is the exact argmax. Probe P25
compares it with edgeR (`probes/out/p25_end_to_end.txt`):

| Dataset | Filter mismatches | TMM factors | AveLogCPM | φ | a (emulated / edgeR) | s2, median rel. diff | d0 (emulated / edgeR) | log10 p, max abs diff | FDR < 0.1 (emulated / edgeR / both) |
|---|---|---|---|---|---|---|---|---|---|
| kang | 0 | 4e-15 | 4e-12 | 2e-15 | 1 / 1 | 1.6e-4 | 9.15406 / 9.15334 | 2.9e-3 | 695 / 695 / 695 |
| sim3 | 0 | 4e-15 | 7e-12 | 3e-15 | 1 / 1 | 5.2e-4 | 6.73481 / 6.73541 | 2.9e-3 | 187 / 187 / 187 |
| sim3b | 0 | 4e-15 | 4e-12 | 3e-15 | 1 / 1 | 5.2e-4 | 5.5418 / 5.54184 | 2.9e-3 | 165 / 165 / 165 |
| sim8 | 0 | 5e-15 | 4e-12 | 2e-16 | 1.589183 / 1.589321 | 2.2e-4 | 11.8186 / 11.8169 | 6.3e-3 | 262 / 262 / 262 |
| airway | 0 | 3e-15 | 4e-12 | 3e-15 | 1 / 1 | 3.2e-4 | 5.38802 / 5.38793 | 1.5e-3 | 6568 / 6568 / 6568 |
| thyroid32 | 0 | 5e-15 | 7e-12 | 2e-15 | 1.852336 / 1.852844 | 4.4e-4 | 4.09614 / 4.09621 | 4.6e-3 | 1388 / 1387 / 1387 |

The TMM factor and φ columns are maximum relative differences; the AveLogCPM
column is the maximum absolute difference. A few boundary genes in airway
differ more in s2 (maximum 0.51), but their s2.post agrees to 9e-4.

## 13. Reference outputs

These are produced by `scripts/reference_edger.R` and live in
`validation/work/edger/<dataset>/`.

**Files**:

- **`genes_ql.tsv`**: one row per kept gene, with
  - `gene`, the 0-based row of counts.mtx
  - `AveLogCPM`, `logFC`, `logCPM`, `F`, `PValue`, `FDR`
  - `coefficient` (shrunk, natural log) and `unshrunk_coefficient` for the
    tested column
  - `deviance` (at φ/a), `df_residual` (n − p)
  - `deviance_adj`, `df_residual_adj`, `s2` (= deviance_adj / df_residual_adj)
  - `s2_prior`, `s2_post`, `df_prior`, `df_total`, `iter`, `failed`
- **`genes_lrt.tsv`**: estimateDisp, glmFit and glmLRT output: `logFC`,
  `logCPM`, `LR`, `PValue`, `FDR`, the coefficients, `trended_dispersion`,
  `tagwise_dispersion` and `deviance`.
- **`filter.tsv`**: `kept` and the row total for every gene.
- **`samples.tsv`**: raw library sizes, library sizes after filtering, TMM
  factors and effective library sizes.
- **`coefficients_ql.tsv`**: every shrunk coefficient (natural log).
- **`reference.json`**:
  - QL: φ, top.proportion, average.ql.dispersion, prior count, df.prior,
    discoveries.
  - LRT: common dispersion, span, prior.df, prior.n, discoveries.
- **`timing.json`**: seconds per step, the QL total, the LRT total and the BLAS
  in use.
- **`timing-r2/` and `timing-r3/`**: repeat runs.
- **Unit matrices**, written for airway, kang, sim3, sim3b, sim8 and thyroid32
  from an untimed `glmQLFit(keep.unit.mat = TRUE)`: `leverage.tsv`,
  `unit_deviance_adj.tsv`, `unit_df_adj.tsv`, `fitted_values.tsv` and
  `offset.tsv`.

**Tested coefficients**:

| Dataset | Tested coefficient |
|---|---|
| airway | `dextrt` |
| kang | `stim` |
| sim3, sim3b, sim8 | `group` |
| thyroid32, thyroid128, thyroid | `sex[T.female]` |

Each is the last design column. These match scripts/validate_bulk.py and
scripts/benchmark_applications.py.

**Historical reference profiling.** The figures below belong to the
original specification study, not the final 1.4.0 timing collection.
Current release timings and agreements are in [BENCHMARK.md](BENCHMARK.md)
and [METHODS.md](../METHODS.md#quasi-likelihood-differential-expression).

One thread, the minimum of 3 runs,
CPUs 0-7 and 12-15. QL is DGEList through topTags; LRT replaces glmQLFit and
glmQLFTest with estimateDisp, glmFit and glmLRT. DESeq2 discoveries are
padj < 0.1 over all genes, from the existing reference tables. Jaccard indexes
are at FDR 0.1.

| Dataset | n | Genes kept | QL s | of which glmQLFit s | LRT s | of which estimateDisp s | QL discoveries | LRT discoveries | DESeq2 discoveries | Jaccard QL–DESeq2 | Jaccard LRT–DESeq2 | Jaccard QL–LRT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airway | 8 | 16860 / 22369 | 0.454 | 0.371 | 1.519 | 1.358 | 6568 | 6697 | 4834 | 0.724 | 0.712 | 0.945 |
| kang | 16 | 999 / 1000 | 0.085 | 0.069 | 0.185 | 0.160 | 695 | 710 | 635 | 0.903 | 0.886 | 0.973 |
| sim3 | 6 | 3142 / 4000 | 0.060 | 0.043 | 0.166 | 0.147 | 187 | 271 | 239 | 0.760 | 0.640 | 0.684 |
| sim3b | 6 | 3214 / 4000 | 0.063 | 0.045 | 0.191 | 0.173 | 165 | 251 | 201 | 0.794 | 0.662 | 0.657 |
| sim8 | 16 | 3196 / 4000 | 0.092 | 0.059 | 0.220 | 0.182 | 262 | 275 | 253 | 0.728 | 0.687 | 0.939 |
| thyroid32 | 32 | 28596 / 28769 | 3.716 | 2.879 | 11.009 | 9.054 | 1387 | 1523 | 1615 | 0.740 | 0.840 | 0.831 |
| thyroid128 | 128 | 28681 / 28769 | 11.575 | 8.731 | 30.389 | 24.696 | 830 | 716 | 792 | 0.735 | 0.793 | 0.794 |
| thyroid | 647 | 28764 / 28769 | 50.623 | 37.364 | 129.121 | 104.839 | 7479 | 7215 | 7177 | 0.917 | 0.946 | 0.953 |

QL parameters for each dataset:

| Dataset | φ | a | df.prior |
|---|---|---|---|
| airway | 0.01133 | 1 | 5.39 |
| kang | 0.02334 | 1 | 9.15 |
| sim3 | 0.1260 | 1 | 6.74 |
| sim3b | 0.1381 | 1 | 5.54 |
| sim8 | 0.08792 | 1.589 | 11.8 |
| thyroid32 | 0.1185 | 1.853 | 4.10 |
| thyroid128 | 0.1225 | 1.937 | 3.91 |
| thyroid | 0.1364 | 1.848 | 3.88 |

`summary.json` holds these numbers, plus the Jaccard indexes restricted to the
genes edgeR kept. It is produced by `summarize.py`, and `run_all.sh` reruns
everything.

## Appendix: probe index (`probes/`)

| Probe | Finding |
|---|---|
| p0_components.R | components of the glmQLFit and glmQLFTest objects |
| p1_tmm_filter.R/.py, p1b_tmm_ties.py, p1c_filter_arithmetic.py | TMM trim, tie and arithmetic rules; filterByExpr rule and arithmetic; aveLogCPM; addPriorCount |
| p22_filter_tol.R | filterByExpr tolerance on the minimum sample size (1e-14) |
| p2_nb_dispersion.R | top.proportion = chooseLowessSpan(G√d, 20, 0.02); ceiling(t·G) top genes; Cox-Reid common dispersion; fit$dispersion is not rescaled |
| p23_common_disp.R, p24_apl.R | Brent on φ^(1/4) over [0, 4^(1/4)] with tol 1e-5; APL = loglik − ½ log det(XᵀWX) |
| p3_unit_airway.py, p4_unit_grid.R, p5_unit_h.R | unit df ∝ (1 − h); factors depend on the mean only (at a = 1); no further leverage dependence |
| p6*.R/.py, p7_avg_scan.R | the working dispersion is φ/a; the divided-count form of the moments |
| p8_moment_grid.R, p8_moments.py | edgeR's implied unit-deviance moments against exact moments (table) |
| p9_test_eb.R, p26_logfc.R | squeezeVar equivalence; F, df.total and p formulas; LR at φ/a; prior-count coefficients; AveLogCPM |
| p10-p16*.R/.py | the average.ql.dispersion rule (lowess on s2^(1/4), span 0.5, 0.9 quantile, two passes) and the cases where it diverges |
| p17-p20*.R, p27*.R | fitFDistUnequalDF1: floor, log-moment terms, inverse-trigamma weights, weightedLowess span rule, profile ML for df2; weightedLowess delta |
| p21_trended_route.R | glmQLFit on a DGEList with estimateDisp output: mean of the trended dispersions over the top 10% |
| p25_end_to_end.R with emulate.R and emulate_full.R | end-to-end reproduction (section 12) |
