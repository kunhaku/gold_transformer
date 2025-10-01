# Gold Transformer

The **Gold Transformer** project provides a refactored pipeline for training and evaluating a recurrent transformer model on gold (XAUUSD) market data. The repository packages data ingestion, preprocessing, model training, evaluation, and visualization into composable modules while maintaining backward compatibility with earlier helper scripts.

## Project structure

| Path | Description |
| --- | --- |
| `configs/` | Dataclass definitions for data and model hyperparameters. |
| `data/` | Utilities for ingesting MT5 data, engineering features, constructing sliding windows, and managing datasets. |
| `models/` | Recurrent transformer implementation, loss functions, and the training loop. |
| `evaluation/` | Inference routine that persists predictions to SQLite and computes regression metrics. |
| `pipelines/` | High-level orchestration for end-to-end training. |
| `visual_tool.py` | Dash application for inspecting saved predictions. |
| Legacy helpers (`config.py`, `data_preprocessing.py`, `test_model.py`) | Thin wrappers around the refactored modules for existing scripts. |

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

1. Configure your MetaTrader 5 credentials inside `fetch_XAU.py`.
2. Run the script to download the latest 5-minute XAUUSD candles, store them in `mt5_data.db`, and export a CSV snapshot:

   ```bash
   python fetch_XAU.py
   ```

   The script creates the database and CSV automatically if they do not already exist.

## Training pipeline

1. Ensure the SQLite database specified by `DataConfig.db_path` contains the desired price history (defaults to `mt5_data.db`).
2. Launch the training pipeline:

   ```bash
   python pipeline.py
   ```

   This command loads data, generates rolling windows, trains the recurrent transformer, writes artifacts to the `artifacts/` and `models/` directories, saves predictions to SQLite, and opens the Dash visualization app for inspection.

### Customizing configuration

`DataConfig` and `ModelConfig` define the parameters used throughout the pipeline (e.g., window lengths, feature columns, transformer depth, learning rate). Update their defaults in `configs/data_config.py` and `configs/model_config.py`, or instantiate custom objects in your own scripts:

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
python test_model.py
```

The compatibility wrapper loads the default artifacts and appends a fresh set of predictions to the SQLite database.

## Visualizing results

The training pipeline automatically launches the Dash dashboard to explore predictions versus ground truth. If you need to reopen the visualization later, run:

```bash
python visual_tool.py
```

Pass a custom configuration dictionary if the artifacts live in a different location.

## Troubleshooting

- TensorFlow must be installed with GPU or CPU support compatible with your environment.
- Ensure the SQLite database exists and includes the `open`, `high`, `low`, `close`, and `tick_volume` columns before training.
- Delete or rename the prediction table if you want a clean inference database; `run_inference` recreates the table each time it runs.