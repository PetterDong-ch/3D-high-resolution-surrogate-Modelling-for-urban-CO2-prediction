# Independent Urban 3D CO2 Surrogate Redesign

## 0. Design Goal

The current V40 model is a contextual autoregressive 3D U-Net. It performs well
inside Camden when previous CO2 is available, but it still uses PALM-resolved
high-resolution fields such as local `u`, `v`, `w`, `p`, and `theta`.

The next model should be a more independent surrogate. At inference time it
should not require PALM to first generate high-resolution meteorology. The model
should instead use:

- externally available large-scale meteorological forcing, for example ERA5;
- high-resolution emissions;
- urban morphology and surface descriptors;
- background CO2;
- optional previous CO2;
- time and coordinate information.

The key design rule is:

> Tensor shape does not define information type. A scalar or vertical profile
> broadcast to `[Z,H,W]` is still not a true high-resolution 3D field.

---

## 1. Channel Classification

This table separates variables by their real information structure, not by the
shape they may have after preprocessing.

| Channel | Physical Meaning | Original Dimensionality | Varies in x | Varies in y | Varies in z | Varies across samples/time | Recommended Encoder | Broadcast Required | Potential Redundancy |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `prev_kc_CO2` | CO2 concentration at previous output time | 3D | Yes | Yes | Yes | Yes | Local 3D spatial encoder; optional global low-res encoder | No | Strongly overlaps with target; can dominate prediction |
| `prev_dCdx` | x-gradient of previous CO2 | 3D derived | Yes | Yes | Yes | Yes | Local 3D spatial encoder | No | Redundant with `prev_kc_CO2`, but useful for transport texture |
| `prev_dCdy` | y-gradient of previous CO2 | 3D derived | Yes | Yes | Yes | Yes | Local 3D spatial encoder | No | Redundant with `prev_kc_CO2`, but useful for transport texture |
| `prev_dCdz` | vertical gradient of previous CO2 | 3D derived | Yes | Yes | Yes | Yes | Local 3D spatial encoder | No | Redundant with `prev_kc_CO2`, but useful for vertical mixing tendency |
| `current/historical kc_CO2` | CO2 field from current or historical state | 3D | Yes | Yes | Yes | Yes | Only if physically available at inference | No | Current `kc_CO2(t)` is target leakage if used as input |
| `emission_values` | local/source emission field | TO VERIFY: often 2D surface or 3D injected field | Yes | Yes | TO VERIFY | Yes | Local spatial encoder; global downsampled encoder | Maybe | Strong driver; may duplicate with global emission if both crops are used |
| `building_mask` | building occupancy by voxel | 3D or 2.5D derived | Yes | Yes | Yes if voxelized | Usually static | Local 3D encoder or mask processor | Maybe | Related to building height/topography |
| `fluid_mask` | valid atmospheric cells | 3D mask | Yes | Yes | Yes | Usually static | Local mask input and loss mask | No | Redundant with building/topography but needed for validity |
| `x_norm` | normalized domain x-coordinate | 3D broadcast from x | Yes | No | No | No | Coordinate channel in local/global branch | Broadcast in y,z | May encode absolute location; helps domain memorization |
| `y_norm` | normalized domain y-coordinate | 3D broadcast from y | No | Yes | No | No | Coordinate channel in local/global branch | Broadcast in x,z | May encode absolute location; helps domain memorization |
| `z_norm` | normalized height coordinate | 3D broadcast from z | No | No | Yes | No | Coordinate channel or profile conditioning | Broadcast in x,y | Redundant with explicit height index/profile |
| `building_height` | building height or canopy height | 2D | Yes | Yes | No | Usually static | 2D CNN then lift to 3D, or append to each z with clear label | Yes | Overlaps with voxel building/fluid mask |
| `topography/topo_all` | terrain/building/other topography class | 2D or 2.5D | Yes | Yes | No or derived | Usually static | 2D CNN; categorical embedding/one-hot | Yes | Overlaps with building mask/fluid mask |
| `vegetation_type` | land-cover category | 2D categorical | Yes | Yes | No | Usually static or seasonal TO VERIFY | 2D embedding/one-hot + 2D CNN | Yes | Overlaps with EVI/LSWI |
| `pavement_type` | pavement/road surface category | 2D categorical | Yes | Yes | No | Usually static | 2D embedding/one-hot + 2D CNN | Yes | Overlaps with street type/albedo |
| `street_type` | street or road class | 2D categorical | Yes | Yes | No | Usually static | 2D embedding/one-hot + 2D CNN | Yes | Overlaps with pavement/emissions |
| `water_type` | water surface category | 2D categorical | Yes | Yes | No | Usually static | 2D embedding/one-hot + 2D CNN | Yes | Overlaps with albedo/land cover |
| `albedo_type` | surface albedo category/value | 2D | Yes | Yes | No | Usually static/seasonal TO VERIFY | 2D CNN; categorical embedding if class | Yes | Overlaps with land cover |
| `ERA5_u(z)` | large-scale wind profile, x component | 1D vertical profile | No | No | Yes | Yes | 1D CNN/profile encoder | Yes only for FiLM or late fusion | Do not treat as high-res local wind |
| `ERA5_v(z)` | large-scale wind profile, y component | 1D vertical profile | No | No | Yes | Yes | 1D CNN/profile encoder | Yes only for FiLM or late fusion | Do not treat as high-res local wind |
| `temperature_profile` | atmospheric temperature profile | 1D vertical profile | No | No | Yes | Yes | 1D CNN/profile encoder | Yes only for FiLM or late fusion | Related to stability variables |
| `pressure_profile` | atmospheric pressure profile | 1D vertical profile | No | No | Yes | Yes | 1D CNN/profile encoder | Yes only for FiLM or late fusion | Related to height/background |
| `humidity_profile` | atmospheric moisture profile | 1D vertical profile | No | No | Yes | Yes | 1D CNN/profile encoder | Yes only for FiLM or late fusion | TO VERIFY availability |
| `background_CO2_profile` | incoming/background CO2 by height | 1D profile | No | No | Yes | Yes | 1D CNN/profile encoder or additive baseline | Yes if reconstructing 3D field | May dominate absolute concentration |
| `hour_sin/cos` or `tod_sin/cos` | time-of-day cyclic encoding | scalar pair | No | No | No | Yes | MLP scalar encoder | Broadcast only if FiLM needs spatial map | Redundant with emissions if emissions are time-specific |
| `month_sin/cos` | seasonal cyclic encoding | scalar pair | No | No | No | Yes | MLP scalar encoder | Broadcast only if FiLM needs spatial map | May encode split/month bias |
| `day_of_year_sin/cos` | seasonal time encoding | scalar pair | No | No | No | Yes | MLP scalar encoder | Broadcast only if FiLM needs spatial map | Redundant with month |
| `boundary_layer_height` | boundary-layer depth | scalar | No | No | No | Yes | MLP scalar encoder | No | May summarize vertical mixing |
| `mean_wind_speed` | bulk wind speed | scalar | No | No | No | Yes | MLP scalar encoder | No | Derived from wind profile |
| `wind_direction` | bulk wind direction | scalar/cyclic pair | No | No | No | Yes | MLP scalar encoder | No | Derived from wind profile |
| `scalar_background_CO2` | background CO2 scalar | scalar | No | No | No | Yes | MLP scalar encoder or additive baseline | Maybe | Redundant with profile background |
| `domain_id` | city/domain identity | categorical scalar | No | No | No | No per domain | Embedding if multi-domain training | No | Can overfit to domain |

### Important Current-Code Distinction

In V40, `u`, `v`, `w`, `p`, and `theta` are PALM-resolved high-resolution
fields. In the proposed independent surrogate, they should be replaced by
ERA5/profile/scalar forcing. They should not be silently reused unless the
target deployment still has PALM-resolved fields available, which would defeat
the stated goal.

---

## 2. Channel-to-Branch Mapping

| Channel Group | Local Spatial Branch | Global Spatial Branch | Forcing Branch | Notes |
|---|---:|---:|---:|---|
| Local high-res emissions | Yes | Optional downsampled full-domain version | No | Local controls nearby sources; global controls remote/upwind sources |
| Full-domain emissions | No | Yes | No | Use low-res full-domain map, not duplicate local crop only |
| Building/fluid mask | Yes | Optional downsampled version | No | Local mask is essential; global mask helps domain-scale blockage/context |
| Building height/topography | Yes via 2D surface encoder | Optional downsampled version | No | Treat as 2D/2.5D, not true 3D unless voxelized |
| Land-use categories | Yes via embedding/one-hot + 2D CNN | Optional downsampled version | No | Do not use raw arbitrary integer categories as continuous physics |
| `x_norm`, `y_norm`, `z_norm` | Yes | Optional for global location | No | Useful but can encourage domain memorization |
| Previous CO2 local crop | Yes for Task A | Optional low-res full-domain previous CO2 | No | Strong baseline; must compare with persistence |
| Previous CO2 gradients | Yes for Task A | Usually no | No | Local texture/transport state |
| Background CO2 profile | No, unless reconstructed as baseline map | Optional if full-domain background varies spatially TO VERIFY | Yes | Prefer forcing/profile branch or additive baseline |
| ERA5 wind profile | No | No | Yes | Do not broadcast as fake 3D wind unless using FiLM modulation |
| Temperature/pressure/humidity profiles | No | No | Yes | Profile encoder |
| Boundary-layer height, wind speed, wind direction | No | No | Yes | Scalar MLP |
| Time encodings | No direct spatial meaning | No | Yes | MLP, then condition spatial branch |
| Domain ID | No | No | Optional embedding | Only for multi-city training; dangerous for generalization |

Recommended responsibility split:

```text
Local   = high-resolution emissions + urban geometry + local previous state
Global  = downsampled full-domain spatial context + remote/upwind source context
Forcing = ERA5/background/time/sample-level atmospheric state
```

---

## 3. Recommended Input Tensor Definitions

### Task A: Autoregressive Model with Previous CO2

Inputs:

```text
X_local:
  shape [B, C_local, Z, H, W]
  includes local emissions, masks, urban geometry, coordinates,
  local prev_kc_CO2 and prev gradients.

X_global:
  shape [B, C_global, Zg, Hg, Wg] or [B, C_global, Z, Hg, Wg]
  includes downsampled full-domain emissions, morphology, masks,
  optional full-domain prev_kc_CO2.

X_profile:
  shape [B, C_profile, Z_profile]
  includes ERA5/background vertical profiles.

X_scalar:
  shape [B, C_scalar]
  includes time, BLH, wind direction/speed, scalar descriptors.
```

Target:

```text
Delta C_t = C_t - C_{t-1}
C_hat_t = C_{t-1} + Delta C_hat_t
```

Key notes:

- Previous CO2 should enter the local branch.
- A low-resolution global previous CO2 can help if local tiles do not contain
  enough upstream plume context.
- The global branch may add less once local previous CO2 is strong.
- Always compare against persistence: `C_hat_t = C_{t-1}`.
- To avoid over-reliance on previous CO2, evaluate delta metrics, active-event
  metrics, sign accuracy, and hard-change cases.

### Task B: Forcing-Only Model without Previous CO2

Inputs:

```text
X_local:
  local emissions, masks, urban geometry, coordinates.

X_global:
  full-domain emissions, full-domain morphology, domain position.

X_profile:
  ERA5/background vertical profiles.

X_scalar:
  time and sample-level forcing descriptors.
```

Target:

```text
Y = C_t - C_background,t
C_hat_t = C_background,t + Y_hat
```

Why harder:

- Without `C_{t-1}`, the model has no direct plume memory.
- It must infer CO2 transport from emissions, urban form and coarse forcing.
- Remote/upwind emissions become much more important.
- ERA5 is coarse and cannot directly describe PALM-scale street-canyon flow.

Most important variables:

- full-domain emissions and upwind/downwind context;
- wind profile/direction/stability;
- background CO2 profile;
- local building/fluid mask and morphology;
- absolute tile coordinates.

---

## 4. Minimal-Change Architecture

This version keeps the current 3D U-Net style and changes only the input
organization and fusion.

### Inputs

```text
X_local  : [B, C_local, 16, 256, 256]
X_global : [B, C_global, 16, 80, 80]       # low-res full-domain context
X_profile: [B, C_profile, Z_profile]
X_scalar : [B, C_scalar]
```

### Architecture

1. Local 3D U-Net encoder-decoder keeps the V40 backbone.
2. Global context encoder remains a compact 3D CNN over low-res full-domain
   maps.
3. New forcing encoder:
   - 1D CNN over vertical profiles;
   - MLP over scalar variables;
   - concatenate both into `F_forcing`.
4. Fuse `F_forcing` at bottleneck:
   - safest implementation: project to bottleneck channels and add/broadcast;
   - better implementation: FiLM modulation per scale.
5. Decoder outputs either:
   - Task A: `pred_delta`;
   - Task B: `pred_enhancement = C_t - C_background,t`.

### Checkpoint Compatibility

This is partially compatible with V40:

- local U-Net weights can be loaded;
- global encoder can be partially reused if global input channels are similar;
- first convolution likely needs reinitialization because PALM high-res
  meteorology channels are removed/replaced;
- forcing encoder and FiLM layers are new and must be trained from scratch.

---

## 5. Scientifically Cleaner Multimodal Architecture

This version uses separate encoders for separate information structures.

### Local Spatial Encoder

Inputs:

```text
true_3d_local_fields: emissions if 3D, fluid/building voxel mask,
                      previous CO2, previous gradients, coordinates
surface_2d_fields   : building height, topography, land use, water, street type
```

Handling:

- use 3D CNN for true 3D local fields;
- use 2D CNN/embedding for surface fields;
- lift 2D surface features to 3D by height-aware broadcasting or learned
  vertical lifting.

### Global Spatial Encoder

Inputs:

```text
downsampled full-domain emissions
downsampled morphology/masks
optional full-domain previous CO2
domain/tile position grid
```

Purpose:

- remote-source information;
- upwind/downwind plume context;
- domain-scale background;
- long-range spatial dependency.

If input is the complete `800 x 800` domain, this branch may become redundant.
In that case, run an ablation:

```text
full-domain model with explicit global branch
vs
full-domain model without explicit global branch
```

Do not keep the branch by default if it duplicates information.

### Forcing Encoder

Inputs:

```text
ERA5 profiles: [B, C_profile, Z_profile]
scalars      : [B, C_scalar]
```

Encoder:

```text
profile_embedding = 1D_CNN(X_profile)
scalar_embedding  = MLP(X_scalar)
forcing_embedding = MLP(concat(profile_embedding, scalar_embedding))
```

Fusion:

- recommended first implementation: bottleneck concatenation/addition;
- stronger version: FiLM conditioning at each encoder/decoder scale:

```text
h_s' = gamma_s(F_forcing) * h_s + beta_s(F_forcing)
```

---

## 6. Forward-Pass Pseudocode

```text
surface_feat_2d = SurfaceEncoder2D(surface_2d)
surface_feat_3d = LiftTo3D(surface_feat_2d, z_grid)

local_input = concat(true_3d_local, surface_feat_3d)
local_features = LocalUNetEncoder(local_input)

global_features = GlobalEncoder3D(global_lowres)
global_local = sample_global_to_local(global_features, tile_grid)

profile_feat = ProfileEncoder1D(profile_forcing)
scalar_feat = ScalarMLP(scalar_forcing)
forcing_feat = FusionMLP(concat(profile_feat, scalar_feat))

for each U-Net scale:
    local_features[scale] = FiLM(local_features[scale], forcing_feat)

bottleneck = concat(local_bottleneck, global_local_bottleneck, forcing_feat_broadcast)
decoded = UNetDecoder(bottleneck, skip_connections)

if Task A:
    pred_delta = OutputHead(decoded)
    pred_CO2 = prev_CO2 + pred_delta

if Task B:
    pred_enhancement = OutputHead(decoded)
    pred_CO2 = background_CO2 + pred_enhancement
```

---

## 7. PyTorch Module Skeleton

```python
class ProfileEncoder1D(nn.Module):
    def __init__(self, in_channels, hidden=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x_profile):
        x = self.net(x_profile).squeeze(-1)
        return self.proj(x)


class ScalarEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_scalar):
        return self.net(x_scalar)


class FiLM3D(nn.Module):
    def __init__(self, feature_channels, cond_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * feature_channels)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        shape = [x.shape[0], x.shape[1], 1, 1, 1]
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)
        return (1.0 + gamma) * x + beta


class IndependentCO2Surrogate(nn.Module):
    def __init__(self, local_channels, global_channels, profile_channels, scalar_dim):
        super().__init__()
        self.surface_encoder = SurfaceEncoder2D(...)       # TO IMPLEMENT
        self.local_encoder = Local3DUNetEncoder(...)       # TO IMPLEMENT
        self.global_encoder = GlobalContextEncoder3D(...)  # can adapt V40
        self.profile_encoder = ProfileEncoder1D(profile_channels)
        self.scalar_encoder = ScalarEncoder(scalar_dim)
        self.forcing_fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.film_blocks = nn.ModuleList([
            FiLM3D(32, 256),
            FiLM3D(64, 256),
            FiLM3D(128, 256),
            FiLM3D(256, 256),
        ])
        self.decoder = Local3DUNetDecoder(...)             # TO IMPLEMENT
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x_local_3d, x_surface_2d, x_global, x_profile, x_scalar,
                tile_grid=None, prev_co2=None, background_co2=None):
        surface = self.surface_encoder(x_surface_2d)
        surface_3d = lift_surface_to_3d(surface, x_local_3d.shape[-3:])
        local_input = torch.cat([x_local_3d, surface_3d], dim=1)

        skips, bottleneck = self.local_encoder(local_input)
        global_feat = self.global_encoder(x_global)
        global_local = sample_or_pool_global(global_feat, tile_grid, bottleneck.shape[-3:])

        profile_feat = self.profile_encoder(x_profile)
        scalar_feat = self.scalar_encoder(x_scalar)
        forcing = self.forcing_fusion(torch.cat([profile_feat, scalar_feat], dim=1))

        conditioned_skips = [film(skip, forcing) for film, skip in zip(self.film_blocks[:-1], skips)]
        bottleneck = self.film_blocks[-1](bottleneck, forcing)
        bottleneck = torch.cat([bottleneck, global_local], dim=1)

        decoded = self.decoder(bottleneck, conditioned_skips)
        residual = self.head(decoded)

        if prev_co2 is not None:
            pred = prev_co2 + residual
        elif background_co2 is not None:
            pred = background_co2 + residual
        else:
            pred = residual
        return {"pred_residual": residual, "pred_co2": pred}
```

---

## 8. Ablation Plan

| Ablation | Inputs | Scientific Question | Expected Interpretation |
|---|---|---|---|
| Local only | local emissions + urban geometry + coordinates | Can local static/source information explain the field? | Baseline for spatial source/morphology skill |
| Local + ERA5 forcing | local + profile/scalar forcing | Does coarse meteorology improve prediction? | If improved, ERA5 adds transport/mixing information |
| Local + Global | local + downsampled full-domain maps | Does remote/upwind source context matter? | Improvement means tile-only input is insufficient |
| Local + ERA5 + Global | all non-history inputs | Best forcing-only spatial surrogate without previous CO2 | Tests full independent surrogate |
| Local + ERA5 + Global + previous CO2 | all inputs with previous CO2 | Best autoregressive model | Must compare to persistence |
| Local + ERA5 + Global without previous CO2 | no history | Can model work when previous state is unavailable? | Key deployment test if no previous CO2 exists |
| Full-domain direct without explicit global branch | full 800x800 local/global merged | Is global branch redundant when full domain is input? | If similar performance, remove global branch |
| Previous CO2 only / persistence | previous CO2 only or delta=0 | How much skill comes from temporal persistence? | Required reference for Task A |

Metrics:

- reconstructed CO2 MAE/RMSE/R;
- delta MAE/RMSE/R;
- active-event sign accuracy;
- high-amplitude delta RMSE;
- per-height and per-month metrics;
- visual comparison on fixed cases.

---

## 9. Data Preprocessing Changes

1. Stop treating ERA5/profile/scalar variables as high-resolution 3D fields.
2. Store four explicit input groups:
   - local 3D crop;
   - local 2D surface crop;
   - global low-res full-domain maps;
   - profile/scalar forcing arrays.
3. Encode categorical land-use variables as one-hot or embeddings.
4. Keep normalization statistics training-only at the job/time level, not just
   cache-label level.
5. Store tile-to-global alignment metadata:
   - local crop origin `(x0,y0,z0)`;
   - full-domain grid size;
   - global sampling grid for `grid_sample`;
   - physical grid spacing.
6. For Task A, ensure `prev_kc_CO2` is strictly `t-1`, not current time.
7. For Task B, reconstruct from background CO2 and evaluate concentration.
8. Store metadata showing which variables are available at inference time.

---

## 10. Risks and Limitations

- ERA5 forcing may be too coarse to resolve street-canyon flow.
- A forcing-only model may produce smoother fields than an autoregressive model.
- Previous CO2 can produce high direct R even when delta prediction is weak.
- Global branch may duplicate information for full-domain direct input.
- Coordinates can improve Camden metrics but hurt transfer if the model
  memorizes location-specific patterns.
- Categorical land-use fields require careful encoding; raw integer values can
  introduce false ordering.
- If background CO2 is very strong, absolute concentration metrics may hide
  poor enhancement/delta structure.
- If validation shares very similar timestamps or overlapping locations with
  training, metrics may overstate generalization.

---

## 11. Unresolved Items To Verify

These must be checked in the dataset files, preprocessing metadata, dataset
classes or checkpoint configuration before implementation.

| Item | Status | Where To Check |
|---|---|---|
| Exact ERA5 variables available | TO VERIFY | raw ERA5/preprocessed forcing files |
| ERA5 vertical levels and units | TO VERIFY | forcing NetCDF/metadata |
| Whether emissions are 2D surface or true 3D | TO VERIFY | cache manifest and raw emission variable dimensions |
| Whether background CO2 is scalar, profile, or spatial field | TO VERIFY | `ls_forcing_right_CO2` files and sidecar builder |
| Exact PALM output time interval | TO VERIFY | job output timestamps |
| Whether `u,v,w,p,theta` in current cache are PALM-resolved | Confirmed from V40 usage, but variable source should be documented | V40 train script and PALM files |
| Global full-domain input shape for new model | TO VERIFY | memory/GPU constraints and desired 800x800 experiment |
| Whether full-domain direct model fits on GPU | TO VERIFY | GPU memory test |
| Category encoding for land-use descriptors | TO VERIFY | current preprocessing and variable meanings |
| Whether Richmond has matching morphology/topography variables | TO VERIFY | Richmond raw data directory |
| Train/val/test split independence at job-time level | TO VERIFY | cache metadata and split builder |
| Whether previous CO2 is allowed in final deployment | TO VERIFY | project/scientific requirement |

---

## 12. Practical Recommendation

Build two next models, not one:

### Model 1: Autoregressive Independent Surrogate

```text
local:   emissions + morphology + masks + coordinates + prev_CO2
global:  downsampled emissions + morphology + optional prev_CO2
forcing: ERA5 profiles + background profile + scalar time/weather
target:  CO2(t) - CO2(t-1)
```

This is the most likely to preserve sharp spatial structure.

### Model 2: Forcing-Only Background-Enhancement Surrogate

```text
local:   emissions + morphology + masks + coordinates
global:  downsampled emissions + morphology
forcing: ERA5 profiles + background CO2 + scalar time/weather
target:  CO2(t) - background_CO2(t)
```

This is scientifically cleaner for deployment if previous CO2 is unavailable,
but it is harder and should be judged against absolute concentration,
enhancement and visual plume-pattern metrics.

The two models answer different questions and should not be collapsed into one
experiment too early.
