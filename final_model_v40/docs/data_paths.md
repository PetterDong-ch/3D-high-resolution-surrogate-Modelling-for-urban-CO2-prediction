# V40 External Data Contract

## Availability

The source data are 2019 PALM NetCDF outputs for the London Camden domain. They
are too large for GitHub and must be obtained from Linfeng Li
([l.li20@imperial.ac.uk](mailto:l.li20@imperial.ac.uk)). Access and reuse remain
subject to the data owner's conditions. The repository contains code and
metadata contracts, not the PALM archive itself.

## Portable Layout

`scripts/prepare_camden_data.sh` expects the following input and creates the
remaining entries:

```text
external_data/
  raw/JOBS/                                  # supplied PALM NetCDF jobs
  patch_cache_v33_camden_z001_010_no1112/    # generated local patches
  coordinate_sidecar_v33_camden_z001_010_no1112/
  v33_context_sidecar_camden_z001_010_no1112/
  v33_previous_co2_sidecar_camden_z001_010_no1112/
  v33_texture_sidecar_camden_z001_010_no1112/
```

The topography file is expected as
`raw/JOBS/cam07_175vm_topo_surf_N02.000.nc`; override it with
`TOPOGRAPHY_PATH` if supplied separately.

## Deterministic Preparation Configuration

| Item | Value |
|---|---|
| Included months | 1-10 |
| Excluded months | 11-12 |
| Train/validation patches | 16,384 / 2,048 |
| Patch shape | 16 x 256 x 256 |
| Vertical patch start | 0-3 |
| Horizontal candidate stride | 32 |
| Time bin size | 8 |
| Split modulo / validation value | 10 / 0 |
| Cache seed | 3300 |
| Global context size | 16 x 80 x 80 |
| Supervised global levels | 1-10 |

The cache builder performs deterministic sample assignment and writes split
manifests. Context, coordinate, previous-CO2, and texture sidecars are aligned
to those manifest rows. Do not mix sidecars generated from another cache.

## Variables and Target

The source archive must expose the PALM variables used by the frozen readers,
including `kc_CO2`, emissions, `ls_forcing_right_CO2`, `u`, `v`, `w`, `p`,
`theta`, and topography/static descriptors. The target is
`kc_CO2(t)-kc_CO2(t-1)` and reconstruction uses the observed previous field.

Buildings, terrain, non-fluid cells, PALM fill values, and non-finite values are
excluded from loss and metrics. The exact variable aliases and fallback rules
are implemented in the frozen scripts under `reproduction/scripts/`.

## Verification

After preparation, run:

```bash
python scripts/verify_package.py --data-root external_data
```

The verifier checks the required manifests, normalization files, metadata, and
texture statistics. Generated cache files should not be committed to Git.
