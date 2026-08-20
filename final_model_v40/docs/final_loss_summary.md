# V40 Loss Summary

V40 predicts the increment:

```text
delta_CO2 = kc_CO2(t) - kc_CO2(t-1)
```

The final objective is a masked composite loss over valid fluid cells and
supervised near-surface levels.

```text
L = L_huber
  + 0.03  L_gradient
  + 0.04  L_global_correlation
  + 0.05  L_local_correlation
  + 0.05  L_amplitude
  + 0.35  L_active_delta
  + 0.04  L_low_frequency
  + 0.02  L_low_frequency_correlation
  + 0.04  L_high_frequency
  + 0.015 L_sign_margin
  + 0.02  L_active_aux
  + 0.02  L_sign_aux
```

Main intent:

- `L_huber`: robust pointwise delta regression.
- `L_gradient`: preserve plume boundaries and local texture gradients.
- `L_global_correlation`: align whole-patch delta pattern.
- `L_local_correlation`: align local 32x32 texture windows.
- `L_amplitude`: avoid under-amplified, over-smooth delta fields.
- `L_active_delta`: upweight cells with meaningful `|delta_CO2|`.
- `L_low_frequency`: supervise broad transport structure.
- `L_high_frequency`: supervise residual texture after low-pass removal.
- `L_sign_margin`: encourage correct positive/negative delta direction.
- `L_active_aux`: train the active-event logit.
- `L_sign_aux`: train the sign logit.

The composite training loss is an optimization signal, not the final physical
metric. Final reporting should use reconstructed concentration and increment
metrics in ppm.
