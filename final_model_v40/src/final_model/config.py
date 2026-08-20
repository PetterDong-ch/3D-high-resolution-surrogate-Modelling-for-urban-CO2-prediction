from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_ROOT = Path(os.environ.get("V40_DATA_ROOT", PROJECT_ROOT / "external_data"))
V33_PROCESSED_ROOT = Path(os.environ.get("V40_PROCESSED_ROOT", EXTERNAL_DATA_ROOT))
CAMDEN_JOBS_ROOT = Path(os.environ.get("CAMDEN_JOBS_ROOT", EXTERNAL_DATA_ROOT / "raw" / "JOBS"))


LOCAL_CHANNELS = [
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "p",
    "theta",
    "w",
    "month_sin",
    "month_cos",
    "tod_sin",
    "tod_cos",
    "x_norm",
    "y_norm",
    "z_norm",
    "prev_kc_CO2",
    "prev_dCdx",
    "prev_dCdy",
    "prev_dCdz",
]

GLOBAL_CHANNELS = [
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "w",
    "p",
    "theta",
    "fluid_mask",
    "prev_kc_CO2_global",
]


# Stores Final V40 settings and paths.
@dataclass(frozen=True)
class FinalV40Paths:
    cache_root: Path = V33_PROCESSED_ROOT / "patch_cache_v33_camden_z001_010_no1112"
    context_sidecar_root: Path = V33_PROCESSED_ROOT / "v33_context_sidecar_camden_z001_010_no1112"
    previous_co2_sidecar_root: Path = V33_PROCESSED_ROOT / "v33_previous_co2_sidecar_camden_z001_010_no1112"
    normalization_root: Path = V33_PROCESSED_ROOT / "v33_context_sidecar_camden_z001_010_no1112"
    metadata_root: Path = V33_PROCESSED_ROOT / "coordinate_sidecar_v33_camden_z001_010_no1112"
    texture_sidecar_root: Path = V33_PROCESSED_ROOT / "v33_texture_sidecar_camden_z001_010_no1112"
    checkpoint: Path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    run_dir: Path = PROJECT_ROOT / "runs" / "v40_keep_prev_monitor"
    eval_dir: Path = PROJECT_ROOT / "evaluations" / "v40_keep_prev_best_model_256_direct_r"


MODEL_CONFIG = {
    "model_variant": "v40_event_texture_context",
    "in_channels": 18,
    "out_channels": 1,
    "base_channels": 32,
    "global_channels": 9,
    "global_feature_channels": 8,
    "context_correction_scale": 1.0,
    "high_delta_scale": 1.0,
    "min_high_gate": 0.20,
    "trainable_parameters": 5761317,
}


DATA_CONFIG = {
    "target_mode": "autoregressive_delta",
    "target_definition": "delta_CO2 = kc_CO2(t) - kc_CO2(t-1)",
    "reconstruction": "pred_CO2(t) = kc_CO2(t-1) + pred_delta_CO2",
    "patch_shape": [16, 256, 256],
    "supervised_global_z": [1, 10],
    "excluded_months": [11, 12],
    "valid_mask": "fluid/topography/missing-value mask",
    "train_samples_per_epoch": 512,
    "validation_samples": 128,
}


LOSS_CONFIG = {
    "base_loss": "masked_huber",
    "huber_delta": 2.0,
    "gradient_loss_weight": 0.03,
    "correlation_loss_weight": 0.04,
    "local_correlation_loss_weight": 0.05,
    "amplitude_loss_weight": 0.05,
    "active_delta_loss_weight": 0.35,
    "low_frequency_loss_weight": 0.04,
    "low_frequency_correlation_weight": 0.02,
    "high_frequency_loss_weight": 0.04,
    "sign_loss_weight": 0.015,
    "active_aux_loss_weight": 0.02,
    "sign_aux_loss_weight": 0.02,
    "residual_weight_alpha": 0.5,
    "residual_weight_scale": 4.0,
    "residual_weight_max": 4.0,
    "target_gradient_weight_alpha": 0.35,
    "target_gradient_weight_scale": 1.0,
    "target_gradient_weight_max": 2.5,
}


TRAIN_CONFIG = {
    "epochs": 70,
    "batch_size": 1,
    "num_workers": 4,
    "learning_rate": 3.0e-5,
    "weight_decay": 1.0e-4,
    "scheduler": "ReduceLROnPlateau",
    "lr_factor": 0.5,
    "lr_patience": 7,
    "min_lr": 1.0e-6,
    "early_stopping_patience": 18,
    "seed": 42,
    "init_weights": str(PROJECT_ROOT / "checkpoints" / "best_model.pt"),
}
