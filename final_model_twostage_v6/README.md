# Two-Stage Camden CO2 Surrogate V6

This directory is the reproducibility release of the final two-stage Camden
experiment. Stage 1 predicts urban meteorological fields from forcing,
geometry, surface and time information. Stage 2 replaces the corresponding
PALM-resolved meteorological channels in the final V40 CO2 surrogate with
Stage-1 predictions and estimates the one-step CO2 increment.

The package contains the model source, preprocessing and evaluation code,
locked dependencies, released checkpoints, compact reference outputs and the
commands required to regenerate the reported results. Large PALM NetCDF files
are not redistributed.

## Reproducibility Status

| Item | Status |
|---|---|
| Model, loss, training and evaluation source | Included |
| Stage 1, Stage 2 and V40-initialisation checkpoints | Included through Git LFS |
| Exact dependency versions | `requirements-lock.txt` and `environment.yml` |
| Seeds and deterministic CUDA settings | Seed 42; deterministic cuDNN enabled |
| Compact metrics and original logs | `artifacts/` |
| Raw PALM simulations | External; see Data Access |
| Generated caches and sidecars | Excluded; regeneration scripts included |

Training on a different GPU, CUDA/cuDNN release or PyTorch build can produce
small numerical differences despite fixed seeds. Reproduction means comparable
scientific results, not necessarily bit-identical floating-point tensors.

## Data Access

The Camden PALM simulations are too large for GitHub and are not public data.
For authorised academic access, contact **Linfeng Li** at
`l.li20@imperial.ac.uk`. Cite the project and request the Camden 2019 PALM jobs
used for this IRP. Access remains subject to the data owner's approval.

After access is granted, arrange the data as:

```text
external_data/
  camden/
    JOBS/
      z##_camden2019##/
        INPUT/
        OUTPUT/
  v40_coordinate_sidecar/
    metadata_manifest.json
    ...
```

The coordinate sidecar can alternatively be regenerated with the companion
`final_model_v40/scripts/prepare_camden_data.sh` workflow. The exact contract is
documented in `docs/data_paths.md`.

## Environment

Checkpoints are larger than GitHub's normal file limit. Clone with Git LFS:

```bash
git lfs install
git clone <repository-url>
cd final_model_twostage_v6
git lfs pull
```

Create the reference Python 3.12 environment:

```bash
conda env create -f environment.yml
conda activate final-model-twostage-v6
python scripts/verify_package.py
```

Alternatively, from a clean Python 3.12 virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python scripts/verify_package.py
```

Do not combine this environment with server-wide `PYTHONPATH` packages; the
historical mixed Python 3.8/3.12 environment is not reproducible.
The compatible clean lock and historical package mismatch are documented in
`docs/environment_provenance.md`.

## Model and Target

Stage 1 is `LocalFNOStage1` with width 32, four spectral blocks and modes
`(8, 16, 16)`. It predicts `u`, `v`, `w` and potential-temperature anomaly
`theta_prime` on a `16 x 256 x 256` grid. The exact channels and loss weights
are frozen in `configs/stage1_fno_camden.yaml`.

Stage 2 is a contextual event-texture 3D U-Net with 18 local channels and a
`9 x 16 x 80 x 80` full-domain context tensor. Its target is:

```text
delta_CO2 = kc_CO2(t) - kc_CO2(t-1)
pred_CO2(t) = kc_CO2(t-1) + pred_delta_CO2
```

Supervision is restricted to global levels 1-10 and valid atmospheric cells.
Stage 1 does not predict pressure, so the Stage-2 pressure channel is the
documented zero placeholder used in the reported experiment.

## Reproduction Order

Run commands from this directory.

1. Verify checkpoints and external data:

   ```bash
   python scripts/verify_package.py --data-root external_data
   ```

2. Build the Stage-1 metadata and cache:

   ```bash
   bash scripts/prepare_stage1_data.sh
   ```

3. Reproduce Stage-1 training (optional when using the released checkpoint):

   ```bash
   bash scripts/reproduce_stage1_training.sh
   ```

4. Build the Stage-2 cache and aligned sidecars using the released Stage-1
   checkpoint:

   ```bash
   bash scripts/prepare_stage2_data.sh
   ```

   To use a newly reproduced Stage-1 checkpoint instead:

   ```bash
   STAGE1_CHECKPOINT="$PWD/generated/stage1_run/best_model.pt" \
     bash scripts/prepare_stage2_data.sh
   ```

5. Reproduce Stage-2 training from the released V40 initialisation:

   ```bash
   bash scripts/reproduce_stage2_training.sh
   ```

6. Evaluate the released Stage-2 checkpoint on the regenerated validation
   population:

   ```bash
   bash scripts/reproduce_report_evaluation.sh
   ```

Set `OVERWRITE=1` only when intentionally replacing generated caches. All
generated outputs are written below `generated/` and are ignored by Git.

## Reference Results

The released Stage-1 checkpoint selected epoch 27 (`best_val_loss=2.0132`). Its
saved physical-unit validation results were:

| Field | MAE | RMSE | R |
|---|---:|---:|---:|
| u | 1.5326 m s-1 | 2.0921 m s-1 | 0.8389 |
| v | 1.7691 m s-1 | 2.3452 m s-1 | 0.8678 |
| w | 0.4285 m s-1 | 0.6536 m s-1 | 0.1761 |
| theta_prime | 0.2204 K | 0.2663 K | 0.9002 |
| reconstructed theta | 0.2282 K | 0.2776 K | 0.9887 |

The released Stage-2 checkpoint selected the best validation-loss state from a
110-epoch run initialised from V40. Evaluation covered 1,988 validation samples:

| Quantity | R | R2 | MAE (ppm) | RMSE (ppm) |
|---|---:|---:|---:|---:|
| Reconstructed CO2 | 0.9590 | 0.9197 | 1.8310 | 3.3335 |
| Delta CO2 | 0.5881 | 0.3367 | 1.8310 | 3.3335 |

Exact unrounded values and checkpoint hashes are in
`reproducibility_manifest.json`.

## Repository Contents

```text
checkpoints/       released weights (Git LFS)
configs/           frozen data, model, loss and training settings
src/               importable Stage-1 and Stage-2 implementation
runtime/           frozen V40 loss/runtime code used by Stage 2
scripts/           preprocessing, training, evaluation and verification
artifacts/         compact reference metrics, histories and original logs
docs/              model and external-data documentation
```

The report source (`.tex`/`.bib` or `.docx`) and figure-generation source must
also be committed at the repository level. They are deliberately not presented
as part of this model-only subdirectory.
