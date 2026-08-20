from __future__ import annotations


STAGE2_TO_V40_CHANNEL = {
    "emission_values": "emission_values",
    "ls_forcing_right_CO2": "ls_forcing_right_CO2",
    "u": "stage1_u",
    "v": "stage1_v",
    "theta": "stage1_theta",
    "w": "stage1_w",
    "month_sin": "month_sin",
    "month_cos": "month_cos",
    "tod_sin": "tod_sin",
    "tod_cos": "tod_cos",
    "x_norm": "x_norm",
    "y_norm": "y_norm",
    "z_norm": "z_norm",
}

V40_STAGE1_MET_CHANNELS = (
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
)

V40_STAGE1_GLOBAL_CONTEXT_CHANNELS = (
    "emission_values",
    "ls_forcing_right_CO2",
    "u",
    "v",
    "w",
    "p",
    "theta",
    "fluid_mask",
    "prev_kc_CO2",
)
