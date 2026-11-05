# Gaussian cis-QTL permutation performance

RapidGLM **1.6.0** measurements on development tag `v1.6.0-dev.2` (`4505ef6afaa00a38c5c871ae0e12d531c91ec1cf`). All 24 stages passed the timing audit; the release library is unchanged from the measured dev.2 source.

Both Gaussian headline comparisons use 10,000 permutations:

| Scope | RapidGLM seconds | tensorQTL seconds | tensorQTL / RapidGLM |
| --- | --- | --- | --- |
| Chromosome 22 | 35.86 | 107 | 2.983× |
| Genome-wide | 1212 | 3069 | 2.531× |


## Workload and timing scope

Geuvadis has 462 samples and 42 covariates plus intercept, with a 1 Mb window and seed 20261016. Chromosome 22 has 580 genes; the genome-wide scan has 20,427 expressed autosomal genes. Each command uses twelve Zen 3 cores (CPUs 6–17), one thread per core, with siblings idle in the exclusive Slurm allocation.

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5. Exclusive Slurm allocation; schedutil governor, boost enabled and SMT enabled, unchanged. Each timed command starts after two ten-second quiet samples. More than 5% other CPU activity on selected cores or any sibling invalidates the command. Host load, other-user processes and vLLM CPU use are retained per command. From the resumed bulk stage on 2026-11-03, a user-authorized runtime service cpuset confines vLLM to CPUs 0-5,22-29,46-47; the reservation receipt records its application and restoration. Earlier accepted measurements retain their original guarded host conditions.

Whole-command chromosome-22 timings:

| Tool | Model | Permutations | Wall seconds | Start load | Peak RSS (GiB) |
| --- | --- | --- | --- | --- | --- |
| rapidglm_gaussian | Gaussian, single precision (default) | 1000 | 4.91 | 3.73 | 0.6504 |
| rapidglm_nb | NB score | 1000 | 498.06 | 4.11 | 1.01 |
| rapidglm_gaussian10000 | Gaussian, single precision (default) | 10000 | 35.86 | 6.62 | 0.6528 |
| rapidglm_gaussian10000_double | Gaussian, double precision | 10000 | 62.53 | 6.74 | 0.856 |
| fastqtl | Gaussian | adaptive 1000-10000 | 528.07 | 6.86 | 0.08057 |
| tensorqtl | Gaussian | 10000 | 106.96 | 6.26 | 1.428 |
| jaxqtl_nb_score_spa_acat | NB SPA+ACAT | 0 | 42.43 | 6.55 | 0.7591 |
| jaxqtl_nb_score_perm1000 | NB score | 1000 | 2257 | 6.64 | 0.9184 |

FastQTL uses adaptive 1,000–10,000 permutations; jaxQTL SPA+ACAT uses a different calibration. Their work budgets differ from the fixed Gaussian headline. Peak RSS is a per-process maximum, rather than aggregate process-tree memory for multiprocess tools.

## Implementation

Gaussian permutations project covariates out of genotypes once per gene, score shuffled-response batches with the packed matrix kernel and reuse worker buffers. Single-precision permutation scores are the default; `permutation_precision: double` retains double arithmetic. Observed statistics, generator streams and Beta calibration retain their definitions. [METHODS](../METHODS.md#cis-qtl-mapping) records equations, precision bounds and optimization alternatives.

## Numerical comparison

The chromosome-22 double-precision 10,000-permutation table is byte-identical to the 1.4.0 reference. Single precision preserves statuses, lead variants, observed statistics and eGene identities, with these measured changes:

| Permutations | Changed empirical p-values | Count-change range | Largest relative beta_p difference | Same eGene identities as 1.4.0 |
| --- | --- | --- | --- | --- |
| 1000 | 12 | 1–1 | 3.7881e-05 | True |
| 10000 | 76 | 1–3 | 2.274e-05 | True |

The genome-wide chromosome-22 rows match this release's chromosome-22 single-precision run byte-for-byte. The genome-wide receipt separately checks eGene identities, statuses, leads and observed statistics against 1.4.1. The chromosome-22 numerical bound is not asserted as a genome-wide bound. Full agreement is in [EQTL_COMPARE.json](../validation/EQTL_COMPARE.json) and [EQTL_GENOME_WIDE.json](../validation/EQTL_GENOME_WIDE.json).

## Reproducibility

The previous 1.4.0/1.4.1 receipts used a different workstation. Compare reference/tool ratios measured on the same host; changes in absolute seconds do not isolate code changes.

[QTL_FINAL.json](../validation/QTL_FINAL.json), the [executed runner copy](../validation/receipts/v1.6.0/run_accepted.sh), [source manifest](../validation/receipts/v1.6.0/source.json) and [timing audit](../validation/receipts/v1.6.0/TIMING_AUDIT.json) retain commands, memory, hashes, accepted attempts and CPU-policy evidence. The GPU-host native, CLI, Python/R binding and container checks passed in both division modes; their records are in [TESTS.json](../validation/TESTS.json). The runbook records the prior full functional suite on this source. The coordinator’s separate full ARM64 run is outside this receipt collection.
