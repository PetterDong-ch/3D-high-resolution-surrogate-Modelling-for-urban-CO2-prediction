# V40 Camden CO2 Surrogate: Reproducibility Package

This directory is the research artifact for the final Camden spatial surrogate
reported as V40. It contains the final and initialization checkpoints, frozen
source used for the reported experiment, a readable standalone API, pinned
dependencies, data-building commands, and reference outputs.

The model predicts the one-step CO2 increment

```text
delta_CO2(t) = kc_CO2(t) - kc_CO2(t-1)
pred_kc_CO2(t) = kc_CO2(t-1) + pred_delta_CO2(t)
```

for 16 x 256 x 256 volumes, with supervision restricted to global vertical
levels 1-10. January-October Camden simulations are used; November and
December are excluded.

## Reproducibility Status

| Requirement | Included here |
|---|---|
| Final source and checkpoint | Yes |
| Exact V38 initialization checkpoint | Yes |
| Frozen preprocessing, training and evaluation source | Yes, under `reproduction/` |
| Readable standalone implementation | Yes, under `src/final_model/` |
| Exact training/evaluation commands | Yes, under `scripts/reproduce_*` |
| Fixed seeds and stochasticity note | Yes |
| Pinned Python dependencies | Yes |
| Checkpoint hashes and expected metrics | Yes |
| Raw PALM data | External; see [Data Access](#data-access) |
| Written-report source | Must be committed at the parent repository level |

The frozen `reproduction/` runtime is authoritative for regenerating the
reported experiment. The smaller `src/final_model/` implementation is intended
for inspection, checkpoint loading, and convenient follow-on experiments; it
must not be substituted when claiming exact reproduction of the report result.

## Data Access

The PALM NetCDF simulation archive and generated caches are too large for
GitHub and are not redistributed in this package. Researchers requesting the
detailed Camden data should contact **Linfeng Li
([l.li20@imperial.ac.uk](mailto:l.li20@imperial.ac.uk))** and state that the
request concerns the 2019 London Camden PALM CO2 simulations used by this IRP.

After access is granted, place or link the raw jobs at:

```text
external_data/raw/JOBS/
```

Alternatively set `CAMDEN_JOBS_ROOT` to their location. Expected filenames,
generated data layout, filters, and provenance are documented in
[`docs/data_paths.md`](docs/data_paths.md).

## Environment

The reference environment is Python 3.12.3. Create a clean environment rather
than reusing system packages:

```bash
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python scripts/verify_package.py
```

`environment.yml` provides an equivalent Conda entry point. CUDA/PyTorch builds
may need to be installed from the platform-specific PyTorch index before the
remaining requirements. Checkpoints are tracked with Git LFS because ordinary
GitHub file storage is not suitable for model weights.
The cleanup of an incompatible historical NumPy/SciPy combination is documented
in `docs/environment_provenance.md`.

## Reproduce the Reported Result

Run these commands from the repository root and in this order.

1. Build the deterministic 16,384/2,048 patch cache and all aligned sidecars:

   ```bash
   bash scripts/prepare_camden_data.sh
   ```

2. Reproduce training from the supplied V38 initialization:

   ```bash
   bash scripts/reproduce_camden_training.sh
   ```

3. Reproduce the 40-sample, 256 x 256 report evaluation using the released
   final checkpoint:

   ```bash
   bash scripts/reproduce_camden_report_evaluation.sh
   ```

To keep external data elsewhere:

```bash
export V40_DATA_ROOT=/absolute/path/to/v40_external_data
export CAMDEN_JOBS_ROOT=/absolute/path/to/JOBS
bash scripts/prepare_camden_data.sh
```

The released evaluation is expected to produce:

| Samples | Valid cells | R | R2 | MAE (ppm) | RMSE (ppm) |
|---:|---:|---:|---:|---:|---:|
| 40 | 13,859,072 | 0.971093 | 0.940498 | 2.465107 | 4.294319 |

Small floating-point differences are acceptable across GPU models and PyTorch
builds. A materially different sample count, mask count, or metric indicates a
data-layout, checkpoint, or environment mismatch.

## Package Verification

Check model hashes and, once data are present, the expected external layout:

```bash
python scripts/verify_package.py --data-root external_data
bash scripts/check_checkpoint.sh
```

Reference SHA-256 values and expected metrics are stored in
`reproducibility_manifest.json`.

## Architecture and Inputs

The local tensor is `[B, 18, 16, 256, 256]` and contains emissions, incoming
background CO2, PALM-resolved `u`, `v`, `p`, `theta`, `w`, cyclic month/time
encodings, normalized coordinates, previous CO2, and its three spatial
gradients. The global tensor is `[B, 9, 16, 80, 80]` and contains emissions,
background CO2, `u`, `v`, `w`, `p`, `theta`, a fluid mask, and full-domain
previous CO2.

`V40EventTextureContextUNet3D` combines a local 3D U-Net, a full-domain context
encoder aligned to each tile by `grid_sample`, a context-correction head, an
event-texture residual branch, and active-event/sign auxiliary heads. The model
has 5,761,317 trainable parameters. The exact composite loss configuration is
recorded in [`docs/final_loss_summary.md`](docs/final_loss_summary.md) and in
the training command.

## Randomness and Hardware Variability

Seeds are fixed (`training=42`, cache generation `3300`) and deterministic
cuDNN behavior is requested in the standalone trainer. Sampling, CUDA kernels,
thread scheduling, library versions, and hardware may still prevent bitwise
identity. Reproduction should therefore be assessed by the same manifests,
sample counts, qualitative fields, and statistically equivalent metrics rather
than identical checkpoint bytes after retraining.

## Directory Guide

```text
checkpoints/       released V40 and V38 initialization weights
reproduction/      frozen original preprocessing/training/evaluation runtime
src/final_model/   readable standalone model API
scripts/           ordered workflows and verification
configs/           recorded final configuration
artifacts/         reference history, log, and metrics (provenance only)
docs/              data contract and method details
```

Historical absolute paths inside `artifacts/` and
`configs/v40_final_config_hpc_provenance.json` are provenance from the original
HPC run. The default JSON config and portable scripts use repository-relative
paths and environment variables.

## Report Source Requirement

This model package does not contain the written report source. The final Git
repository must also include the report's editable source (`.tex` and `.bib`,
or `.docx`), figure/table generation source, and a documented build command.
Committing only the PDF does not satisfy full project reproducibility.
