# Model Summary

Two-stage V6 is built to test whether PALM-resolved meteorological fields can be
replaced by learned meteorology while keeping the final V40 CO2 predictor.

## Stage1: Local-FNO Meteorological Predictor

Stage1 uses a Local-FNO-style network:

- 2D surface encoder lifts surface descriptors into 3D.
- 1D profile encoder summarizes vertical forcing.
- scalar encoder summarizes sample-level descriptors.
- 3D spectral blocks learn spatially structured meteorological fields.

Outputs:

```text
u, v, w, theta
```

## Stage2: V40-Style CO2 Delta Predictor

Stage2 predicts:

```text
delta_CO2 = kc_CO2(t) - kc_CO2(t-1)
```

and reconstructs:

```text
pred_CO2(t) = kc_CO2(t-1) + pred_delta_CO2
```

It uses the V40 contextual event-texture 3D U-Net:

- local 3D U-Net branch
- full-domain low-resolution context branch
- low-frequency context correction
- high-frequency event-texture residual
- active-event and sign auxiliary heads
