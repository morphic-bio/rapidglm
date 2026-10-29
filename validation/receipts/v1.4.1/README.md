# RapidGLM 1.4.1 patch validation

Version 1.4.1 fixes a zero-column weighted cross-product that bound a reference
into empty matrix storage. The existing Cook's replacement-refit regression
exposed this under UBSan when every coefficient was at a boundary. The
cross-product now returns its empty result before forming row pointers.

[Local validation](SANITIZER_FIX.json) records the tested patch commit and
source hashes, before the release version bump:

- The original undefined behavior was reproduced under UBSan.
- Six empty-design cases (0, 1, 3, 4, 5 and 11 rows) pass.
- All 28 tested nonempty-design cross-products are bit-identical to 1.4.0.
- All 29 focused solver/bulk cases and 4,553 assertions pass ASan/UBSan with
  OpenMP, including the existing Cook's replacement-refit regression.

The full release CI results, including both sanitizer configurations and ARM64,
are available in [GitHub Actions](https://github.com/morphic-bio/rapidglm/actions).
The added regression brings the full C++ suite to 173 cases.

The benchmark tables and [raw receipts](../v1.4.0/README.md) remain the completed
1.4.0 campaign on `f0ecfbd7c8dba698bbb50db5fd61785a2a463800`, tagged
`v1.4.0-dev.4`. No benchmark has been remeasured or relabelled as 1.4.1. The
published 1.4.0 tags remain unchanged; its GitHub release is held as a draft
because its sanitizer CI failed. Version 1.4.1 supersedes that release.
