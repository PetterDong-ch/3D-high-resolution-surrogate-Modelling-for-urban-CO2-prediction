# Environment Provenance

The historical HPC environment mixed system and user-installed binary
packages. In particular, it contained `NumPy 2.4.6` with `SciPy 1.11.4`, even
though SciPy 1.11.4 requires `NumPy <1.28`. This was not a clean, resolvable
environment and previously caused pandas/matplotlib import failures.

The reproducibility release pins `NumPy 1.26.4`, satisfying SciPy 1.11.4 and
the other declared packages. Historical metrics and logs are retained as
provenance. As with GPU stochasticity, minor numerical variation relative to
the historical mixed installation should be reported rather than hidden.
