# Developer and Reproduction Usage

The authoritative workflow is in `README.md`. In brief:

```bash
python scripts/verify_package.py --data-root external_data
bash scripts/prepare_stage1_data.sh
bash scripts/reproduce_stage1_training.sh
bash scripts/prepare_stage2_data.sh
bash scripts/reproduce_stage2_training.sh
bash scripts/reproduce_report_evaluation.sh
```

Every shell entry point accepts `PYTHON_BIN`; data locations can be overridden
with `CAMDEN_JOBS_ROOT`, `V40_COORDINATE_SIDECAR_ROOT` and the variables shown
inside the corresponding script. Generated data are written below `generated/`.
