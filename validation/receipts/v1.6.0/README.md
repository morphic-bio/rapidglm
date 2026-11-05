# RapidGLM 1.6.0 receipts

Measured library: `4505ef6afaa00a38c5c871ae0e12d531c91ec1cf` (`v1.6.0-dev.2`). The runner and documents are separate commits on the private development branch. The release library is unchanged from the measured dev.2 source.

One AMD Threadripper PRO 5965WX workstation (Zen 3). Timed work uses one thread per physical core on CPUs 6-21, with SMT siblings 30-45 idle; one thread uses CPU 6, eight use 6-13, twelve use 6-17. Drivers and untimed checks use CPUs 0-5. Exclusive Slurm allocation; schedutil governor, boost enabled and SMT enabled, unchanged. Each timed command starts after two ten-second quiet samples. More than 5% other CPU activity on selected cores or any sibling invalidates the command. Host load, other-user processes and vLLM CPU use are retained per command. From the resumed bulk stage on 2026-11-03, a user-authorized runtime service cpuset confines vLLM to CPUs 0-5,22-29,46-47; the reservation receipt records its application and restoration. Earlier accepted measurements retain their original guarded host conditions.

## Built artifacts

- `rapidglm:1.6.0`: `sha256:3aa4599e9b04782e03fe0830c3ded6257cccd8886d24729b709c5b3cb932d4e5`
- `rapidglm-solvers:1.6.0`: `sha256:056e31358744c2d079f7950a9f12bf11eb94f1e908d55b880718521c78b7e740`

Native CLI SHA-256: `fbb618a5a537e04bdafb44351a0791967c9b49e6324cba95194439eec444fbdd`.

## GPU-host quick checks

Native ctest and CLI tests, binding checks, and container tests passed in both division modes, untimed on CPUs 0–5. [TESTS.json](TESTS.json) retains commands, exits and logs; [SMOKE.json](SMOKE.json) retains the successful supervised host/container smoke. The wider functional suite was already completed separately on the measured commit.

## Stage audit

| Stage | Exits | Accepted commands | Start load range | Policy samples | Command reruns |
|---|---|---:|---|---:|---:|
| glmacc | 0 | 0 | — | 177 | 0 |
| glmspeed | 0 | 408 | 1.826–6.693 | 11396 | 3 |
| batch | 1, 0 | 15 | 2.525–4.588 | 3249 | 0 |
| solvers | 0 | 6 | 3.175–4.270 | 809 | 0 |
| solvers_control | 0 | 3 | 4.192–4.752 | 227 | 0 |
| bulkval | 0 | 60 | 0.044–4.870 | 3852 | 0 |
| edgerval | 0 | 0 | — | 15 | 0 |
| bulk | 143, 0 | 198 | 2.255–6.726 | 9907 | 14 |
| cooks | 0 | 0 | — | 289 | 0 |
| rust | 0 | 8 | 4.364–6.751 | 220 | 0 |
| staging | 0 | 0 | — | 650 | 0 |
| pert50k | 0 | 14 | 4.656–6.801 | 3846 | 1 |
| pertfull | 0 | 7 | 5.593–6.866 | 2996 | 0 |
| seeds | 0 | 0 | — | 1412 | 0 |
| untestable | 0 | 12 | 5.152–6.747 | 901 | 0 |
| crt | 0 | 6 | 5.394–6.882 | 2231 | 0 |
| replogle | 0 | 83 | 4.164–6.700 | 9631 | 0 |
| qtl | 0 | 8 | 3.733–6.860 | 3845 | 0 |
| special | 0 | 6 | 4.491–6.691 | 1732 | 0 |
| vmath_e2e | 0 | 42 | 4.025–4.831 | 1387 | 0 |
| ablations | 0 | 5 | 4.413–6.793 | 4808 | 0 |
| replogle_full | 0 | 4 | 6.523–6.871 | 28056 | 0 |
| eqtl_gw | 0 | 2 | 6.485–6.522 | 4383 | 0 |
| compress | 0 | 0 | — | 372 | 0 |

[TIMING_AUDIT.json](TIMING_AUDIT.json) contains the complete policy, quiet-start and contention checks. [STAGE_RERUNS.json](STAGE_RERUNS.json) preserves discarded attempts and their archive hashes.

The first batch attempt was discarded after a redundant outer driver measurement included internal quiet waits. [DRIVER_SCOPE_REPAIR.json](DRIVER_SCOPE_REPAIR.json) records the controller-only correction, its regression tests and the unchanged library, binaries, images and thresholds. All batch commands were measured again; [BATCH_RERUN_INPUT_CHECKS.json](BATCH_RERUN_INPUT_CHECKS.json) verifies identical regenerated inputs. [ABLATION_SCOPE_REPAIR.json](ABLATION_SCOPE_REPAIR.json) records the preventive correction of the Replogle ablation wrapper before that stage ran; its timing lock and inner guards were retained, with no accepted timing invalidated.

## Agreement and ablations

[AGREEMENT_JOB_687.json](AGREEMENT_JOB_687.json) compares the retained earlier collection. Historical timings from that incomplete 1.5.0 run are not paper measurements. [RUN_RECORD_FIELDS.json](RUN_RECORD_FIELDS.json) records the optimization choices and counters emitted by the new runs. [PREPARED_INPUT_CHECKS.json](PREPARED_INPUT_CHECKS.json) verifies all 449 prepared input copies and their retained source files after the run.

| Ablation | Default seconds | Opt-out seconds | Opt-out / default | Agreement |
|---|---:|---:|---:|---|
| replogle_strict | 1058.060 | 1321.300 | 1.249× | passed |
| replogle_ks | 226.900 | 521.140 | 2.297× | passed |
| crt_t100_g7418_ks | 645.885 | 1012.671 | 1.568× | passed |
| crt_t100_g7418_default | 672.998 | 1040.960 | 1.547× | passed |
| qtl_nb_chr22 | 498.067 | 541.492 | 1.087× | passed |

[OPTIMIZATION_ABLATIONS.json](OPTIMIZATION_ABLATIONS.json) retains input pairing, output hashes and agreement checks.

## Ratios and scope

[HEADLINE_COMPARISON.json](HEADLINE_COMPARISON.json) gives reference-tool/RapidGLM ratios and their changes from 1.4.0. Each ratio uses measurements on one host; cross-host seconds are not compared as speed gains.

The separate AWS Gasperini run and its receipt are managed independently. No cloud compatibility claim is made by this GPU-host run. The development optimization receipts remain unchanged.

Release publication was separately authorized on 2026-11-05 after this collection passed its acceptance checks. Historical receipts retain the decisions and scope recorded when they were created.


[RUST_RECONSTRUCTED_COMPARISON.json](RUST_RECONSTRUCTED_COMPARISON.json) compares the new tables with an explicitly labelled untimed reconstruction from the preserved, hash-verified historical binaries. The reconstruction also matches all retained historical scientific diagnostics.

## Historical Rust evidence

The user accepted exact retained historical scientific diagnostics plus byte comparison against a documented untimed reconstruction from the preserved historical binaries. The original historical raw tables are unavailable; byte identity with those deleted original tables remains unestablished.
