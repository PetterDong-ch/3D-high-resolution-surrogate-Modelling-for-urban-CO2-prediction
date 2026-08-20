# Evaluation-Only Use

Obtain the authorised Camden PALM data from `l.li20@imperial.ac.uk`, create the
locked environment from `environment.yml`, and follow Steps 1, 2 and 4 in the
main `README.md` to regenerate the aligned evaluation inputs. Then run:

```bash
bash scripts/reproduce_report_evaluation.sh
```

The released Stage-1 and Stage-2 checkpoints are loaded from `checkpoints/`.
Outputs are written to `generated/report_evaluation/`. The expected metrics are
recorded in `reproducibility_manifest.json`.
