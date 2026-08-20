# Task2 V3 Height-Resolved Mean CO2 Surrogate

This directory is the reproducibility release of the final Task2 profile model.
It predicts the horizontal-mean CO2 enhancement at 16 vertical levels using
current-time inputs only. It is a compact 1-D profile surrogate, not a spatial
3-D field model.

## Scientific Definition

The target and reconstruction are:

```text
enhancement(t,z) = mean_kc_CO2(t,z) - mean_background_CO2(t,z)
pred_mean_CO2(t,z) = mean_background_CO2(t,z) + pred_enhancement(t,z)
```

No previous-timestep CO2 channel is used. The model concatenates 30 local
profile features and 16 global/context profile features, then predicts one
16-level enhancement profile with `CompactProfileNet`.

## Reproducibility Status

| Item | Status |
|---|---|
| Model, loss, training and evaluation source | Included |
| Camden and Richmond fine-tuned checkpoints | Included through Git LFS |
| Exact dependencies | `requirements-lock.txt` and `environment.yml` |
| Randomness control | Seed 123; deterministic cuDNN enabled |
| Profile-cache construction | Included in `scripts/build_average_profile_cache.py` |
| Raw PALM and spatial intermediate data | External; see Data Access |
| Compact reference metrics/histories | `artifacts/` |

Different GPU and CUDA/cuDNN versions can produce small numerical differences.
The expected standard is comparable metrics under the documented workflow,
rather than bit-identical training output.

## Data Access

The Camden and Richmond PALM NetCDF data are too large for GitHub and are not
redistributed. For authorised academic access, contact **Linfeng Li** at
`l.li20@imperial.ac.uk`, identifying the IRP and the requested PALM domains.
Data release remains subject to the owner's approval.

For evaluation, place the supplied or regenerated compact profile caches at:

```text
external_data/
  camden_average_profile_cache/
    manifest.json
    normalization.json
    train_profiles.npz
    val_profiles.npz
  richmond_average_profile_cache/
    manifest.json
    normalization.json
    train_profiles.npz
    val_profiles.npz
```

To regenerate these caches, see `docs/data_paths.md` and
`scripts/prepare_profile_cache.sh`.

## Environment

```bash
git lfs install
git lfs pull
conda env create -f environment.yml
conda activate final-model-task2-v3
python scripts/verify_package.py
```

From a clean Python 3.12 virtual environment, `pip install -r
requirements-lock.txt` is an equivalent alternative. Avoid server-wide
`PYTHONPATH` overlays because mixed binary packages can invalidate the runtime.

## Reproduction and Evaluation

Run commands from this directory.

Verify the released package and caches:

```bash
python scripts/verify_package.py --data-root external_data
```

Evaluate the released Camden model:

```bash
bash scripts/evaluate_camden_task2_v3.sh
```

Reproduce Camden training:

```bash
bash scripts/train_camden_task2_v3.sh
```

Reproduce Richmond transfer learning from the Camden checkpoint:

```bash
bash scripts/train_richmond_finetune_task2_v3.sh
```

Evaluate the released Richmond fine-tuned model:

```bash
bash scripts/evaluate_richmond_task2_v3.sh
```

Generated runs and evaluations are ignored by Git. Override any cache with
`PROFILE_CACHE_ROOT=/path/to/cache` without changing source code.

## Model and Loss

Input shapes:

```text
local features:  [B, 30, 16]
global features: [B, 16, 16]
output:          [B, 16]
```

Training minimises weighted Huber loss in normalized enhancement space plus a
vertical-gradient term:

```text
loss = weighted_huber(pred_enhancement, target_enhancement)
     + 0.05 * vertical_gradient_huber
```

Layer weights emphasize low and low-validity levels. Exact target means,
standard deviations, valid fractions and weights are frozen in `configs/`.

## Reference Results

| Domain/checkpoint | R | R2 | MAE (ppm) | RMSE (ppm) | Values |
|---|---:|---:|---:|---:|---:|
| Camden | 0.9931 | 0.9862 | 0.7854 | 1.2311 | 28,111 |
| Richmond fine-tuned | 0.9832 | 0.9668 | 1.0760 | 1.7452 | 4,276 |

These are concentration metrics after denormalizing the enhancement and adding
the current incoming background profile. Exact values and checkpoint SHA-256
hashes are in `reproducibility_manifest.json`.

## Repository Contents

```text
checkpoints/       released Camden and Richmond weights (Git LFS)
configs/           frozen normalization and loss metadata
src/task2_v3/      modular model, data, loss, metrics and training code
scripts/           cache preparation, train, evaluate and verification entry points
artifacts/         compact reference results and histories
docs/              model and external-data documentation
```

The report source (`.tex`/`.bib` or `.docx`) and figure-generation source must
be committed at repository level in addition to this model-only package.
