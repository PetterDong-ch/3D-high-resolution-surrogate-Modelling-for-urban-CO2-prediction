# Developer and Reproduction Usage

The authoritative workflow, data contract, exact metrics and environment are in
`README.md`. The principal commands are:

```bash
python scripts/verify_package.py --data-root external_data
bash scripts/evaluate_camden_task2_v3.sh
bash scripts/train_camden_task2_v3.sh
bash scripts/train_richmond_finetune_task2_v3.sh
bash scripts/evaluate_richmond_task2_v3.sh
```

Use `scripts/prepare_profile_cache.sh` only when rebuilding profile caches from
the V40-compatible spatial cache and sidecars.
