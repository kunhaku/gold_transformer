from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import tensorflow as tf

from data.datasets import SequenceDataset
from data.scaling import get_target_scaler, inverse_transform, load_scaler_metadata
from evaluation.metrics import compute_regression_metrics
from utils.io import ensure_directory


@dataclass(frozen=True)
class ForecastRecord:
    """Container holding a single autoregressive prediction."""

    group_id: int
    sample_idx: int
    step: int
    valid_length: int
    prediction: np.ndarray
    target: np.ndarray


def _resolve_target_scaler(
    *, scaler_metadata: dict | None, scaler_metadata_path: str | None
) -> dict | None:
    """Load scaler metadata if it was not explicitly provided."""

    if scaler_metadata is None:
        scaler_metadata = load_scaler_metadata(scaler_metadata_path)
    return get_target_scaler(scaler_metadata)


def iterate_group_predictions(
    model: tf.keras.Model,
    dataset: SequenceDataset,
    *,
    scaler_metadata: dict | None = None,
    scaler_metadata_path: str | None = None,
) -> Iterator[ForecastRecord]:
    """Yield predictions for every sample while respecting group order."""

    forecast_length = dataset.targets.shape[1]
    target_scaler = _resolve_target_scaler(
        scaler_metadata=scaler_metadata, scaler_metadata_path=scaler_metadata_path
    )

    unique_groups = np.unique(dataset.group_ids)
    for group_id in unique_groups:
        indices = np.where(dataset.group_ids == group_id)[0]
        past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

        for step, idx in enumerate(indices):
            x_i = dataset.inputs[idx][None, ...]
            y_i = dataset.targets[idx][None, ...]
            m_i = dataset.input_mask[idx][None, ...]
            y_mask_i = dataset.target_mask[idx][None, ...]

            preds = model(x_i, mask=m_i, past_preds=past_preds, training=False)
            past_preds = preds

            pred_values = preds.numpy().flatten()
            true_values = y_i.flatten()

            if target_scaler is not None:
                pred_values = inverse_transform(pred_values, target_scaler)
                true_values = inverse_transform(true_values, target_scaler)

            valid_length = int(np.sum(y_mask_i))
            yield ForecastRecord(
                group_id=int(group_id),
                sample_idx=int(idx),
                step=int(step),
                valid_length=valid_length,
                prediction=pred_values,
                target=true_values,
            )


def generate_group_forecast(
    model: tf.keras.Model,
    dataset: SequenceDataset,
    group_id: int,
    *,
    scaler_metadata: dict | None = None,
    scaler_metadata_path: str | None = None,
) -> List[ForecastRecord]:
    """Return all autoregressive predictions for *group_id*.

    The helper reuses the same sequential roll-out as :func:`run_inference`
    but stops once the requested group has been processed. This enables higher
    level orchestration components (e.g., the revisit supervisor) to retrieve
    consistent traces without recomputing the entire dataset.
    """

    records: List[ForecastRecord] = []
    for record in iterate_group_predictions(
        model,
        dataset,
        scaler_metadata=scaler_metadata,
        scaler_metadata_path=scaler_metadata_path,
    ):
        if record.group_id == group_id:
            records.append(record)
        elif records:
            break
    return records


def run_inference(
    model: tf.keras.Model,
    dataset: SequenceDataset,
    db_path: str,
    table_name: str = "model_b_inference",
    *,
    scaler_metadata: dict | None = None,
    scaler_metadata_path: str | None = None,
) -> dict:
    """Run autoregressive inference on the dataset and persist results to SQLite."""

    ensure_directory(db_path)
    forecast_length = dataset.targets.shape[1]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    y_columns = ", ".join([f"y_{i+1} REAL" for i in range(forecast_length)])
    p_columns = ", ".join([f"p_{i+1} REAL" for i in range(forecast_length)])
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INT,
            sample_idx INT,
            valid_length INT,
            {y_columns},
            {p_columns}
        )
        """
    )

    placeholders = ", ".join(["?"] * (3 + 2 * forecast_length))
    insert_sql = (
        f"INSERT INTO {table_name} (group_id, sample_idx, valid_length, "
        f"{', '.join([f'y_{i+1}' for i in range(forecast_length)])}, "
        f"{', '.join([f'p_{i+1}' for i in range(forecast_length)])}) "
        f"VALUES ({placeholders})"
    )

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for record in iterate_group_predictions(
        model,
        dataset,
        scaler_metadata=scaler_metadata,
        scaler_metadata_path=scaler_metadata_path,
    ):
        row = [record.group_id, record.sample_idx, record.valid_length]
        row.extend(float(v) for v in record.target)
        row.extend(float(v) for v in record.prediction)
        cursor.execute(insert_sql, row)

        all_preds.append(record.prediction)
        all_targets.append(record.target)

    conn.commit()
    conn.close()

    metrics = {}
    if all_preds:
        metrics = compute_regression_metrics(np.vstack(all_targets), np.vstack(all_preds))
    return metrics
