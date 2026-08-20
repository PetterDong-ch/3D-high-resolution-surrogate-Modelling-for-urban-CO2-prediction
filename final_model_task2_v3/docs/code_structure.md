# Code Structure

The clean package separates the profile model into small modules instead of one long runtime file.

## Model And Loss

- `src/task2_v3/network.py`
  - Defines `CompactProfileNet`, the 1D profile network.
- `src/task2_v3/losses.py`
  - Defines the weighted Huber loss and vertical-profile gradient term.
- `src/task2_v3/model.py`
  - Compatibility export layer for common model objects.

## Data And Metrics

- `src/task2_v3/data.py`
  - Loads average-profile cache arrays.
  - Builds the background-enhancement target.
  - Computes target normalisation and vertical layer weights.
- `src/task2_v3/metrics.py`
  - Pearson correlation, R2, MAE, and RMSE.
- `src/task2_v3/io.py`
  - JSON and CSV helpers.

## Training And Evaluation

- `src/task2_v3/engine.py`
  - DataLoader creation and evaluation output writing.
- `src/task2_v3/cli.py`
  - Command-line train/evaluate workflow.
- `src/task2_v3/runtime.py`
  - Thin wrapper kept so existing shell scripts still work.

## Config

- `src/task2_v3/config.py`
  - Central default paths and checkpoint names.
