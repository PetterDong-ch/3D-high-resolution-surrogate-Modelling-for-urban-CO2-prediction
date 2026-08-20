from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
STAGE1_RUNTIME_ROOT = PROJECT_ROOT
DATA_ROOT = Path(os.environ.get("TWOSTAGE_V6_DATA_ROOT", PROJECT_ROOT / "external_data"))
CAMDEN_JOBS_ROOT = Path(os.environ.get("CAMDEN_JOBS_ROOT", DATA_ROOT / "camden" / "JOBS"))


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
    "prev_kc_CO2",
]


STAGE2_CACHE_ROOT = DATA_ROOT / "stage2_cache"
PREVIOUS_CO2_SIDECAR_ROOT = DATA_ROOT / "previous_co2_sidecar"
FULL_DOMAIN_CONTEXT_SIDECAR_ROOT = DATA_ROOT / "full_domain_context_sidecar"
STAGE1_MANIFEST = DATA_ROOT / "stage1_manifest" / "stage1_manifest.json"


MODEL_CONFIG = {
    "stage1": {
        "class": "LocalFNOStage1",
        "geometry_channels": 5,
        "surface_channels": 8,
        "profile_channels": 6,
        "scalar_channels": 4,
        "width": 32,
        "depth": 4,
        "modes_z": 8,
        "modes_y": 16,
        "modes_x": 16,
        "predict_w": True,
        "output_pressure": False,
    },
    "stage2": {
        "class": "V40EventTextureContextUNet3D",
        "in_channels": 18,
        "out_channels": 1,
        "base_channels": 32,
        "global_channels": 9,
        "global_feature_channels": 8,
        "context_correction_scale": 1.0,
        "high_delta_scale": 1.0,
        "min_high_gate": 0.20,
    },
}


DATA_CONFIG = {
    "target_mode": "autoregressive_delta",
    "target_definition": "delta_CO2 = kc_CO2(t) - kc_CO2(t-1)",
    "reconstruction": "pred_CO2(t) = kc_CO2(t-1) + pred_delta_CO2",
    "patch_shape": [16, 256, 256],
    "supervised_global_z": [1, 10],
    "stage1_outputs_used_by_stage2": ["u", "v", "w", "theta"],
    "stage2_pressure_channel": "zero placeholder p, because Stage1 does not predict p",
    "full_domain_context_shape": [9, 16, 80, 80],
}
