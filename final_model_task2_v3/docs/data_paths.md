# External Data Contract

Raw Camden and Richmond PALM simulations are external because of their size and
access restrictions. Authorised users should request the data from
`l.li20@imperial.ac.uk`.

## Evaluation-Ready Contract

Each profile-cache directory must contain:

```text
manifest.json
normalization.json
train_profiles.npz
val_profiles.npz
```

Use `external_data/camden_average_profile_cache` and
`external_data/richmond_average_profile_cache`, or override
`PROFILE_CACHE_ROOT` in the train/evaluation commands.

## Regenerating a Profile Cache

The builder consumes the spatial patch cache and two aligned sidecars generated
by the V40-compatible preprocessing workflow:

```text
external_data/<region>_spatial_patch_cache/
external_data/<region>_context_sidecar/
external_data/<region>_previous_co2_sidecar/
```

For Camden:

```bash
REGION=camden OUT_DIR="$PWD/generated/camden_average_profile_cache" \
  bash scripts/prepare_profile_cache.sh
```

For Richmond:

```bash
REGION=richmond OUT_DIR="$PWD/generated/richmond_average_profile_cache" \
  bash scripts/prepare_profile_cache.sh
```

The profile cache is deterministic given identical source caches. Its manifest
records the selected channels and source paths.
