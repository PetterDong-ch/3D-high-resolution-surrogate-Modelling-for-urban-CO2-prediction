# Code Structure

The clean package separates model code, data adapters, and checkpoint helpers.

## Stage1

- `src/twostage_v6/stage1_fno.py`
  - Defines `LocalFNOStage1`.
  - Predicts Stage1 meteorological fields: `u`, `v`, `w`, `theta`.

## Stage2 Model

- `src/twostage_v6/unet_blocks.py`
  - Reusable 3D convolution, residual, U-Net down/up, and global-context blocks.
- `src/twostage_v6/stage2_model.py`
  - Defines the final Stage2 architecture:
    `V40EventTextureContextUNet3D`.
  - Combines the local 3D U-Net, global-context correction, and event-texture residual heads.
- `src/twostage_v6/model.py`
  - Compatibility export layer.
  - Existing imports from `twostage_v6.model` continue to work.

## Stage2 Data Interface

- `src/twostage_v6/stage2_constants.py`
  - Channel names and V40-compatible channel contracts.
- `src/twostage_v6/stage2_utils.py`
  - Coordinate grids, finite differences, context downsampling, and physical-input assembly.
- `src/twostage_v6/stage2_readers.py`
  - Lightweight shard and sidecar readers.
- `src/twostage_v6/stage2_datasets.py`
  - PyTorch datasets for local-only and full-domain-context Stage2 training/evaluation.
- `src/twostage_v6/stage2_cache.py`
  - Compatibility export layer.
  - Existing imports from `twostage_v6.stage2_cache` continue to work.

## Checkpoints And Config

- `src/twostage_v6/checkpoint.py`
  - Loads saved Stage1 and Stage2 checkpoints.
- `src/twostage_v6/config.py`
  - Central default paths and model constants.
