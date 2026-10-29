# Geuvadis chromosome 22 validation inputs

## 1. Data sources and terms

All inputs are open access. No controlled-access data was used.

The script names below refer to the published copies in the repository.
Set `EQTL_WORK` to the prepared data/tool work directory (default:
`validation/work/eqtl`). `scripts/compare_eqtl.py` uses that directory for
`tools/Rlib` and calls the published `scripts/eqtl_qvalue.R`; ignored scratch
scripts are not needed.

| Data | URL | Terms |
|---|---|---|
| Geuvadis RNA-seq gene quantifications, E-GEUV-1 (ArrayExpress, now BioStudies) | https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/001/E-GEUV-1/Files/E-GEUV-1/analysis_results/ (`GD660.GeneQuantCount.txt.gz`, `GD660.GeneQuantRPKM.txt.gz`, `GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz`, `EUR373/YRI89.gene.cis.FDR5.*`); sample sheet `.../Files/E-GEUV-1.sdrf.txt` | The BioStudies API marks the study public (`isPublic: true`, released 2012-11-06). The IGSR Geuvadis data-reuse statement (http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/geuvadis/README_geuvadis_datareuse_statement.md, saved in `downloads/geuvadis/`) asks users to cite Lappalainen et al. 2013 (Nature 501:506) and refers to the 1000 Genomes reuse terms. EMBL-EBI Terms of Use (https://www.ebi.ac.uk/about/terms-of-use): EMBL-EBI "places no additional restrictions on the use or redistribution of the data" beyond those of the original owners, and expects attribution. |
| Geuvadis genotypes, chr22 (1000 Genomes phase 1 + Omni-imputed, 465 individuals, GRCh37) | `.../Files/E-GEUV-1/genotypes/GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz` | Same as above. |
| 1000 Genomes phase 3, chr22 (2,504 individuals, GRCh37), plus panel and README files | https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ | IGSR disclaimer (https://www.internationalgenome.org/IGSR_disclaimer): "Data from the 1000 Genomes Project is now available without embargo, following the final publication from the project"; users must not infringe third-party rights; EMBL-EBI Terms of Use apply. IGSR FAQ: "The 1000 Genomes data is made available according to the Fort Lauderdale Agreement." The IGSR API lists the phase 3 reuse policy as http://www.1000genomes.org/about#g1k_data_reuse (that anchor now redirects to the general page). Cite the 1000 Genomes Project Consortium 2015 (Nature 526:68) and IGSR (Fairley et al. 2020, NAR 48:D941). Downloaded for planning only; not used in the runs below. |
| GENCODE v12 annotation (the Geuvadis annotation) | https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_12/gencode.v12.annotation.gtf.gz | gencodegenes.org data access page: "All the GENCODE project data is open access". Cite Harrow et al. 2012. |

Checksums: every E-GEUV-1 file used matched the md5 in the BioStudies file list
(`downloads/geuvadis/geuv_filelist.tsv`).

Downloaded so far: about 1.1 GB of data, 0.07 GB of tool sources and binaries, and roughly
1 GB of Python packages.


## 3. Data preparation (chr22)

Scripts: `scripts/eqtl_prep_expression.py` (+ `scripts/eqtl_tmm_factors.R`), `scripts/eqtl_prep_genotypes.sh`,
`scripts/eqtl_make_covariates.py`.

**Samples.** The SDRF lists exactly one library for each of the 462 RNA-seq individuals (the
"QC-passed unique" set): 89 YRI, 91 CEU, 94 GBR, 95 FIN, 93 TSI; 246 female, 216 male. Counts for
these libraries are the matching columns of `GD660.GeneQuantCount.txt.gz`, renamed to the
individual ID. All files use the same sorted order of the 462 IDs (`data/expression/samples.txt`).

**Counts.** `GD660.GeneQuantCount` has 53,934 GENCODE v12 genes. Its values are not all
integers: 26.1% of entries in the 462 columns are non-integer and 10.0% are more than 0.01 from
an integer (fractional read assignment). We round to the nearest integer; this changes total
counts by 0.0025%. `chr22.counts.bed.gz` holds the rounded counts.

**Library size** (`library_size.tsv`, `offset_log_libsize.tsv`): the sum of rounded counts over
all 53,934 genes (all chromosomes), before any gene filter. Range 17.95M to 151.2M, median
54.6M. The jaxQTL offset is log(library size), passed with `--offset` because the chr22 BED is
restricted. `library_size.tsv` also has the TMM factors and the TMM effective library size of
the expressed genes.

**TSS.** The `Coord` column of the Geuvadis files is the strand-aware gene start. It equals the
GENCODE v12 gene TSS for all 20,427 expressed genes (checked against the GTF). The BEDs use
start = TSS - 1 and end = TSS, the convention in the FastQTL and jaxQTL documentation, with
chromosome label `22`.

**Expressed genes** (GTEx v8 thresholds, RPKM in place of TPM): autosomal genes with
RPKM >= 0.1 in >= 20% of samples and rounded count >= 6 in >= 20% of samples. That leaves
20,427 genes genome-wide and 580 on chr22: 362 protein_coding, 101 pseudogene, 33 IG_V_gene,
29 antisense, 24 lincRNA, 17 processed_transcript, and 14 of other types. All gene types are
kept. The same 580 genes go to every tool.

**Normalised expression for FastQTL and the Gaussian models** (GTEx v8 style):
1. TMM factors from edgeR 4.6.2 `calcNormFactors(method = "TMM")` on the expressed-gene counts.
2. CPM with the TMM effective library sizes.
3. A per-gene rank-based inverse normal transform across the 462 samples:
   `qnorm(rank / (n + 1))`, with ties given the average rank.

**Genotypes.** Source: the Geuvadis chr22 release, 493,859 records (474,982 SNP, 18,629 INDEL,
248 SV). The pipeline (bcftools 1.13, `scripts/eqtl_prep_genotypes.sh`) does the following:
1. Adds the missing `##contig` and FORMAT/PP, BD header lines, which bcftools needs to subset
   samples.
2. Keeps the 462 RNA-seq individuals and biallelic SNVs only.
3. Requires a call in all 462 individuals: 391,610 SNVs remain. 42 Omni-imputed individuals
   (41 of them in the RNA set) are unphased and missing at about 19% of sites, so without this
   step each tool would apply its own missing-data rule.
4. Keeps MAF >= 0.05 in the 462: 89,287 SNVs, with no duplicate positions.
5. Strips INFO and every FORMAT field except GT, so that FastQTL (which prefers DS) and jaxQTL
   (PLINK hard calls) read identical genotypes.
6. Sets the ID to `22_POS_REF_ALT_b37`.

PLINK 1 bed/bim/fam and PLINK 2 pgen come from plink2 v2.0.0-a.7.13LM (15 Oct 2026). A1 = ALT,
A2 = REF, FID = 0, IID = sample. Sample order matches `samples.txt` (checked).

**Covariates** (42 in total; `make_covariates.py`):
- Genotype PCs 1-5. plink2 `--indep-pairwise 50 5 0.2` gives 8,071 pruned SNVs, then
  `--pca 10`; the eigenvalues are 41.6, 3.89, 3.56, 2.44, 2.23, then flat at about 2.0. PC1
  separates YRI from the European samples.
- Expression PCs 1-36, from an SVD of the inverse-normal matrix of all 20,427 expressed autosomal
  genes (centred and scaled). K = 36 is a Buja-Eyuboglu parallel analysis: 20 column-permuted
  replicates, 95th percentile. The 36 PCs explain 63% of the variance.
- Sex (female = 1).

This follows the PCA-for-PEER choice (Zhou et al. 2022, Genome Biol 23:210). GTEx v8 used
5 genotype PCs, PEER factors and sex; Geuvadis used PEER k = 10 residuals. The same covariates
are used for every model, count and Gaussian. With the intercept the design has full rank (43).

The Geuvadis PEER-normalised file (`GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz`) was
downloaded but is not used: every tool must see the same normalisation and covariates.


## 4. Tools, versions and command lines

All compute ran under `taskset -c 0-7,12-15 nice -n 10`, never on CPUs 16-31. CPUs 8-11 are
offline on this host, so the allowed set is 12 logical CPUs: 6 P-cores with hyper-threading.
During the runs the host was shared: another session's R jobs were running, on 0-7,12-15 early
on and on 16-31 later, with a load average of 10-37. The timings are therefore indicative only.

**FastQTL.** GTEx fork https://github.com/francois-a/fastqtl, commit 59bcdc0 (the commit used
in the GTEx v8 Dockerfile). The binary reports `v2.184_gtex`; the licence is GPL-3.0.
- Built by the GTEx v8 Docker recipe (gtex-pipeline tag `gtex_v8`, `qtl/Dockerfile`, BSD-3):
  build the standalone Rmath from the R source (here R 4.5.1:
  `./configure --with-x=no --with-readline=no --without-recommended-packages`, then `make` in
  `src/nmath/standalone`); export `RMATH=$W/tools/R-4.5.1/src`; run the recipe's
  `sed -i 's/RMATH=/#RMATH/' Makefile`; `mkdir obj; make`.
- Toolchain: gcc 11.4, Boost 1.74, GSL 2.7.1, system BLAS. The build succeeded with only
  compiler warnings. No FastQTL source file was opened; only its README, its bundled HTML
  documentation (`doc/`), `--help` output and build output were read.
- The wrapper scripts (`python/`, `R/`) were not used. Parallelism uses FastQTL's own
  `--chunk k 12`, with 12 processes started by xargs.
- FastQTL writes no header. Column names follow the GTEx v8 eGene file and tensorQTL
  conventions; `runs/fastqtl/permute/chr22.permute.with_header.txt.gz` has them added.

```
# nominal
fastQTL --vcf data/genotypes/chr22.vcf.gz --bed data/expression/chr22.expr_int.bed.gz \
  --cov data/covariates/covariates.fastqtl.txt.gz --window 1e6 --seed 20261016 \
  --chunk K 12 --out chunks/chunk_K.txt.gz --log chunks/chunk_K.log          # K = 1..12
# permutation pass: adaptive 1,000-10,000 permutations, beta approximation
fastQTL ... same ... --permute 1000 10000 --chunk K 12 ...
```

**jaxQTL** 0.3.1 from PyPI (MIT), in `venv/`:
- Python 3.12.14, jax and jaxlib 0.11.2 (CPU wheels, `jax[cpu]`), numpy 2.5.4, equinox 0.13.8,
  optimistix 0.1.0, lineax 0.1.1, genoio 0.4.1, polars 2.0.0. The full list is in
  `tools/venv_freeze.txt`.
- The CLI enables float64 on CPU. Genotypes are read from PLINK 1 (`--bfile`; hard calls,
  A1 = ALT).
- Common arguments for every run:

```
--bfile data/genotypes/chr22 --covar data/covariates/covariates.jaxqtl.tsv --window 1000000 \
--tss-centered --normalize-covar --seed 20261016 --verbose
```

Runs:
- NB, score test, SPA + ACAT: `jaxqtl cis ... --pheno chr22.counts.bed.gz --offset offset_log_libsize.tsv --model nb --test score --spa --acat`
- NB, score test, Beta permutation: `... --model nb --test score --nperm 1000`
- Gaussian, score test, ACAT: `jaxqtl cis ... --pheno chr22.expr_int.bed.gz --model gaussian --test score --acat`
- Gaussian, score test, Beta permutation:
  `... --model gaussian --test score --nperm 1000`, in two variants:
  - (a) the CLI default, which uses a Student-t reference for the Gaussian family;
  - (b) the same CLI with the non-central chi-square reference the CLI uses for count models
    (`scripts/eqtl_jaxqtl_gauss_ncx2.py`). Variant (a) fails; see section 6.
- Nominal passes: `jaxqtl nominal` with NB score + SPA, and with Gaussian score.
- `EQX_ON_ERROR=nan` is set for every run except the first NB SPA+ACAT run. Without it, one
  solver failure aborts the whole process; with it, failures show up as `perm_converged = false`
  or a NaN `pvalue_adj`.
- Layout: each configuration ran as 4 processes, each pinned to 3 logical CPUs
  (`0-2`, `3-5`, `6,7,12`, `13-15`), with the 580 genes split round-robin
  (`--gene-list data/genelists/chr22.partK.txt`). The parquet outputs were then concatenated
  (`scripts/eqtl_run_jaxqtl_split.sh`). The NB SPA+ACAT configuration also ran as one process on all
  12 CPUs.

**tensorQTL** 1.0.10 (BSD-3) with torch 2.14.1+cpu, in a separate `venv-tensorqtl/`, because its
`pandas-plink` dependency needs numpy < 2 and would have downgraded the jaxQTL venv. Same
inputs as FastQTL. `pyarrow` was pinned below 20 so that it works with numpy 1.26. The full
list is in `tools/venv_tensorqtl_freeze.txt`.

```
python -m tensorqtl data/genotypes/chr22 data/expression/chr22.expr_int.bed.gz chr22 \
  --covariates data/covariates/covariates.fastqtl.txt --mode cis --window 1000000 --seed 20261016
# and the same with --mode cis_nominal
```

`cis` mode uses a fixed 10,000 permutations with the Beta approximation; this is the tensorQTL
default and the GTEx setting.

**Other tools:**
- bcftools and htslib 1.13.
- plink2 v2.0.0-a.7.13LM AVX2 (15 Oct 2026; `tools/plink2/`).
- R 4.5.1 with edgeR 4.6.2 and Bioconductor qvalue 2.40.0. qvalue was installed into
  `tools/Rlib`, not the system library.

**q-values** (`scripts/eqtl_qvalue.R`, `scripts/eqtl_summarize.py`):
- `qvalue::qvalue(p)` with defaults (lambda = 0.05..0.95, smoother pi0), with lambda = 0.85 as a
  fallback (not needed). It is applied to each method's gene-level p-value over that method's
  valid genes.
- The gene-level p-value is `pval_beta` for FastQTL and tensorQTL, and `pvalue_adj` for jaxQTL.
- jaxQTL validity follows its post-processing guide: `result_valid & model_converged &`
  finite `pvalue_adj` in [0, 1], `& perm_converged` for permutation runs.
- eGene: q < 0.05. Benjamini-Hochberg counts are also reported.


## 9. Scripts

| Script | Purpose |
|---|---|
| `scripts/eqtl_prep_expression.py` | samples, rounding, library sizes, gene filter, TMM (calls `eqtl_tmm_factors.R`), inverse normal, expression PCs, parallel analysis, chr22 BEDs |
| `scripts/eqtl_prep_genotypes.sh CHR` | VCF filter, PLINK 1 and 2 files, pruning, genotype PCA |
| `scripts/eqtl_make_covariates.py` | covariate tables in both layouts |
| `scripts/eqtl_run_fastqtl.sh nominal\|permute` | FastQTL with `--chunk k 12` |
| `scripts/eqtl_run_jaxqtl.sh NAME ...` | one jaxQTL process (`CPUS`, `SUB`, `JAXQTL_BIN`, `EQX_ON_ERROR` from the environment) |
| `scripts/eqtl_run_jaxqtl_split.sh NAME ...` | 4 x 3-CPU jaxQTL processes plus merge |
| `scripts/eqtl_run_jaxqtl_chain.sh` | the long jaxQTL sequence. It was stopped before its last step (the full Student-t run); the 25-gene subset ran instead. |
| `scripts/eqtl_jaxqtl_gauss_ncx2.py` | jaxQTL CLI with the ncx2 permutation reference for the Gaussian model |
| `scripts/eqtl_run_tensorqtl.sh cis\|cis_nominal` | tensorQTL |
| `scripts/eqtl_qvalue.R`, `scripts/eqtl_summarize.py` | q-values, overlaps, timings, writing `runs/summary/` |


The table uses the published `eqtl_` filenames. Original scratch paths and
exploratory timing records are historical; new receipts identify their own
version and commit.
