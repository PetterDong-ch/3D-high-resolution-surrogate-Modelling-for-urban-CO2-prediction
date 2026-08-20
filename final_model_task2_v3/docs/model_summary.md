# Model Summary

Task2_V3 is a compact profile model for horizontally averaged CO2 prediction.
It is not a 3D spatial model. It predicts one 16-level vertical profile per
sample.

Target:

```text
enhancement(z) = mean_kc_CO2(t,z) - mean_ls_forcing_right_CO2(t,z)
```

Reconstruction:

```text
pred_mean_CO2(t,z) = background_CO2(t,z) + pred_enhancement(z)
```

The model uses current-time features only and does not require `kc_CO2(t-1)`.
