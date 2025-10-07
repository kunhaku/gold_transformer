# Gold Transformer

The **Gold Transformer** project provides a refactored pipeline for training and evaluating a recurrent transformer model on gold (XAUUSD) market data. The repository packages data ingestion, preprocessing, model training, evaluation, and visualization into composable modules while maintaining backward compatibility with earlier helper scripts.

## Project structure

| Path | Description |
| --- | --- |
| `configs/` | Dataclass definitions for data and model hyperparameters. |
| `data/` | Utilities for ingesting MT5 data, engineering features, scaling columns, constructing progressive sliding windows, and managing datasets. |
| `models/` | Recurrent transformer implementation, loss functions, and the training loop. |
| `evaluation/` | Inference routine that persists predictions to SQLite and computes regression metrics. |
| `pipelines/` | High-level orchestration for end-to-end training. |
| `visual_tool.py` | Dash application for inspecting saved predictions. |
| `legacy/` | Archived helpers (old pipelines, MT5 fetch script, batch trainers, config shims). |

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy, pandas, scikit-learn, tqdm
- Dash and Plotly for visualization
- MetaTrader5 package for data extraction (optional, only needed when running `fetch_XAU.py`)

Install dependencies with pip (adjust the TensorFlow build to match your platform):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow pandas numpy scikit-learn tqdm dash plotly MetaTrader5
```

## Data ingestion

1. Configure your MetaTrader 5 credentials inside `legacy/fetch_XAU.py`.
2. Run the script to download the latest 5-minute XAUUSD candles, store them in `mt5_data.db`, and export a CSV snapshot:

   ```bash
   python legacy/fetch_XAU.py
   ```

   The script creates the database and CSV automatically if they do not already exist.

## Training pipeline

1. Ensure the SQLite database specified by `DataConfig.db_path` contains the desired price history (defaults to `mt5_data.db`).
2. Launch the training pipeline:

   ```bash
   python -m pipelines.training_pipeline
   ```

   The command performs the full offline loop:
   - loads raw candles, computes engineered indicators, and scales price-level features (open/high/low/close/Bollinger bands) together with their moving averages using a shared standardisation;
   - generates progressive sliding windows so every sequence yields a cascade of reveal stages (e.g., initial horizon plus each partial revisit);
   - trains the recurrent transformer with the autoregressive loop that feeds prior predictions back into the model, while logging scalars to TensorBoard and monitoring validation loss for early stopping;
   - writes artifacts to `artifacts/` and `models/`; each run is saved under `models/recurrent_transformer/<run-id>` so checkpoints do not collide with active TensorBoard sessions;
   - runs inference on the held-out split, saving predictions to SQLite; and
   - opens the Dash visualisation app for interactive inspection.

### Monitoring with TensorBoard

Training now emits TensorBoard summaries by default:

- Logs are written to `models/logs/<run-id>` (or to `ModelConfig.tensorboard_log_dir` if you override it). The run-id matches the subdirectory used for the saved model checkpoint when `save_unique_subdir=True`.
- Launch TensorBoard from the project root to inspect curves:

  ```bash
  tensorboard --logdir models/logs
  ```

- Customise behaviour through `ModelConfig`:
  - `tensorboard_log_dir`: change or disable logging (`None` to turn it off).
  - `tensorboard_run_name`: supply a static run label (useful for experiments).
  - `save_unique_subdir`: set to `False` if you prefer the pre-existing flat save layout (note: keep it `True` when TensorBoard is tailing previous runs).
  - `early_stopping_patience` / `early_stopping_min_delta`: tweak validation monitoring.

### Customizing configuration

`DataConfig` and `ModelConfig` define the parameters used throughout the pipeline (e.g., window lengths, feature columns, transformer depth, learning rate, TensorBoard options). Update their defaults in `configs/data_config.py` and `configs/model_config.py`, or instantiate custom objects in your own scripts:

```python
from configs import DataConfig, ModelConfig
from pipelines import run_training_pipeline

data_config = DataConfig(db_path="custom.db", train_ratio=0.75)
model_config = ModelConfig(epochs=20, embed_dim=128)
run_training_pipeline(data_config, model_config)
```

## Running inference on a saved model

After training, reuse the stored model and dataset artifacts to regenerate predictions:

```bash
python legacy/test_model.py
```

The compatibility wrapper loads the default artifacts, applies the saved scaler metadata, and appends a fresh set of predictions to the SQLite database.

## Visualizing results

The training pipeline automatically launches the Dash dashboard to explore predictions versus ground truth. If you need to reopen the visualization later, run:

```bash
python visual_tool.py
```

Pass a custom configuration dictionary if the artifacts live in a different location.

## Revisiting forecasts

The decision layer supports auditing how forecasts evolve as more candles become known. Run the dedicated script to orchestrate training plus the revisit supervisor from the CLI without launching Dash:

```bash
python scripts/run_revisit.py --epochs 5
```

This command reuses the training pipeline, then opens a thesis per test sequence and walks through each reveal stage. The resulting summaries contain per-step metrics, prediction deltas versus the initial thesis, and timestamps that can be fed into governance or reporting tools.

## Troubleshooting

- TensorFlow must be installed with GPU or CPU support compatible with your environment.
- Ensure the SQLite database exists and includes the `open`, `high`, `low`, `close`, and `tick_volume` columns before training.
- Delete or rename the prediction table if you want a clean inference database; `run_inference` recreates the table each time it runs.
