from __future__ import annotations

import sqlite3
from typing import List

import numpy as np
import tensorflow as tf

from data.datasets import SequenceDataset
from evaluation.metrics import compute_regression_metrics
from utils.io import ensure_directory


def run_inference(
    model: tf.keras.Model,
    dataset: SequenceDataset,
    db_path: str,
    table_name: str = "model_b_inference",
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

    unique_groups = np.unique(dataset.group_ids)
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for group_id in unique_groups:
        indices = np.where(dataset.group_ids == group_id)[0]
        past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

        for idx in indices:
            x_i = dataset.inputs[idx][None, ...]
            y_i = dataset.targets[idx][None, ...]
            m_i = dataset.input_mask[idx][None, ...]
            y_mask_i = dataset.target_mask[idx][None, ...]

            preds = model(x_i, mask=m_i, past_preds=past_preds, training=False)
            past_preds = preds

            pred_values = preds.numpy().flatten()
            true_values = y_i.flatten()
            valid_length = int(np.sum(y_mask_i))

            row = [int(group_id), int(idx), valid_length]
            row.extend(float(v) for v in true_values)
            row.extend(float(v) for v in pred_values)
            cursor.execute(insert_sql, row)

            all_preds.append(pred_values)
            all_targets.append(true_values)

    conn.commit()
    conn.close()

    metrics = {}
    if all_preds:
        metrics = compute_regression_metrics(np.vstack(all_targets), np.vstack(all_preds))
    return metrics
