# Analysis entrypoints

These 13 source files preserve the original discovery and downstream analysis logic. Full reanalysis requires the relevant public input matrices, downloaded/prepared separately, and optional dependencies in `requirements-full.txt`.

Run from the repository root and consult the entrypoint's `--help`. Current five-cohort external results are verified by `scripts/verify_expanded_external.py`; historical external summaries emitted by the original entrypoints are superseded by the unchanged results in `revision_20261012/expanded_external/`.

Signature/proxy variables are not official deconvolution outputs. The legacy drug `cv_mse` column contains training error.
