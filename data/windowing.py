from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from configs import DataConfig


WindowOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def create_sliding_windows(feature_matrix: np.ndarray, config: DataConfig) -> WindowOutput:
    """Generate padded windows, masks, and group ids from the feature matrix."""

    X_list, y_list, mask_list, group_id_list, y_mask_list = [], [], [], [], []

    input_length = config.initial_input_length
    forecast_length = config.initial_forecast_length
    max_input_length = config.max_input_length
    min_forecast_length = config.min_forecast_length

    total_length = len(feature_matrix)
    start_index = 0
    current_group_id = 0
    last_completed_group_id = -1

    while start_index + input_length + forecast_length <= total_length:
        X = feature_matrix[start_index : start_index + input_length]
        y = feature_matrix[
            start_index + input_length : start_index + input_length + forecast_length, 3
        ]
        mask = np.ones((input_length, 1), dtype="float32")
        y_mask_unpadded = np.ones((forecast_length,), dtype="float32")

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
        group_id_list.append(current_group_id)
        y_mask_list.append(y_mask_unpadded)

        if forecast_length > min_forecast_length:
            input_length += 1
            forecast_length -= 1
        else:
            last_completed_group_id = current_group_id
            current_group_id += 1
            input_length = config.initial_input_length
            forecast_length = config.initial_forecast_length
            start_index += config.initial_input_length

        start_index += 1

    filtered = [
        (X, y, m, gid, ym)
        for X, y, m, gid, ym in zip(
            X_list, y_list, mask_list, group_id_list, y_mask_list
        )
        if gid <= last_completed_group_id
    ]

    if not filtered:
        raise ValueError("No complete sliding windows could be generated with the given configuration.")

    X_filtered, y_filtered, mask_filtered, group_filtered, y_mask_filtered = zip(*filtered)

    X_padded = pad_sequences(
        X_filtered, maxlen=max_input_length, dtype="float32", padding="pre", value=0.0
    )
    mask_padded = pad_sequences(
        mask_filtered, maxlen=max_input_length, dtype="float32", padding="pre", value=0.0
    )
    max_forecast_length = config.initial_forecast_length
    y_padded = pad_sequences(
        y_filtered, maxlen=max_forecast_length, dtype="float32", padding="post", value=0.0
    )
    y_mask_padded = pad_sequences(
        y_mask_filtered, maxlen=max_forecast_length, dtype="float32", padding="post", value=0.0
    )
    group_ids = np.array(group_filtered, dtype=np.int32)

    return X_padded, y_padded, mask_padded, y_mask_padded, group_ids
