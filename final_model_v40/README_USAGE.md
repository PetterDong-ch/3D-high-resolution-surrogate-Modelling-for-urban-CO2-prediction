# Developer and Reproduction Usage

The authoritative reproducibility workflow is in `README.md`. From a clean
Python 3.12 environment, run:

```bash
python scripts/verify_package.py --data-root external_data
bash scripts/prepare_camden_data.sh
bash scripts/reproduce_camden_training.sh
bash scripts/reproduce_camden_report_evaluation.sh
```

The frozen code under `reproduction/` is authoritative for the reported V40
result. The modular implementation under `src/final_model/` is intended for
inspection and follow-on work. Data paths are controlled through
`V40_DATA_ROOT` and `CAMDEN_JOBS_ROOT`; no old experiment directory is required.

For convenient non-report experiments, the standalone wrappers remain:

```bash
bash scripts/train_camden_v40.sh --help
bash scripts/evaluate_camden_v40.sh --help
bash scripts/resume_camden_v40.sh --help
```
