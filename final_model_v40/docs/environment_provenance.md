# Environment Provenance

The historical HPC run used Python 3.12.3 with packages assembled from system
and user-level directories. An audit found `NumPy 2.4.6` alongside
`SciPy 1.11.4`, although SciPy 1.11.4 declares `NumPy <1.28`. The same mixed
environment also exposed an older system Matplotlib and caused binary/API
errors in some evaluation workflows.

The released clean environment therefore pins `NumPy 1.26.4`, which satisfies
SciPy 1.11.4 and the remaining declared dependencies. Model checkpoint loading
and package-level verification were tested after the source cleanup. Small
numerical differences from the historical mixed environment are covered by the
documented stochastic/hardware variability policy; the unrounded historical
metrics remain in `artifacts/` and `reproducibility_manifest.json`.

The archived training entry point also explicitly enables deterministic cuDNN
and seeds all CUDA devices. This is a reproducibility hardening change relative
to the historical execution environment; it does not alter checkpoint loading
or evaluation of the released weights.
