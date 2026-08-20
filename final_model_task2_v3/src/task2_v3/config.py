from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "src/task2_v3"

DATA_ROOT = Path(os.environ.get("TASK2_V3_DATA_ROOT", PROJECT_ROOT / "external_data"))
CAMDEN_PROFILE_CACHE_ROOT = DATA_ROOT / "camden_average_profile_cache"
RICHMOND_PROFILE_CACHE_ROOT = DATA_ROOT / "richmond_average_profile_cache"


MODEL_CONFIG = {
    "class": "CompactProfileNet",
    "local_feature_count": 30,
    "global_feature_count": 16,
    "global_context_enabled": True,
    "base_channels": 64,
    "target": "background_enhancement",
    "target_definition": "enhancement(z) = mean_kc_CO2(t,z) - mean_ls_forcing_right_CO2(t,z)",
    "reconstruction": "pred_mean_CO2(t,z) = incoming_background_CO2(t,z) + pred_enhancement(z)",
    "input_shape": "[B, local+global features, 16]",
    "output_shape": "[B, 16]",
}


LOSS_CONFIG = {
    "base": "weighted_huber_on_normalized_background_enhancement",
    "huber_delta": 1.0,
    "profile_gradient_weight": 0.05,
    "low_layer_alpha": 1.5,
    "low_layer_tau": 4.0,
    "valid_weight_power": 0.25,
    "target_min_std": 0.2,
}
