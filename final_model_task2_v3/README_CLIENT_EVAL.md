# Evaluation-Only Use

After obtaining the authorised profile cache and pulling the Git LFS
checkpoints, create the environment and run:

```bash
python scripts/verify_package.py --data-root external_data
bash scripts/evaluate_camden_task2_v3.sh
# or
bash scripts/evaluate_richmond_task2_v3.sh
```

Outputs are written below `evaluations/`. Expected values are listed in
`reproducibility_manifest.json`.
