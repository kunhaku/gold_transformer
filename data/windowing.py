from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from configs import DataConfig


WindowOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def create_sliding_windows(feature_matrix: np.ndarray, config: DataConfig) -> WindowOutput:
    """Generate padded windows, masks, and group ids from the feature matrix."""

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    y_mask_list: list[np.ndarray] = []
    group_id_list: list[int] = []

    initial_input = config.initial_input_length
    initial_forecast = config.initial_forecast_length
    min_forecast = max(1, config.min_forecast_length)

    total_length = len(feature_matrix)
    group_id = 0
    start_index = 0

    # We advance the starting point by the initial input length once a group is completed.
    while start_index + initial_input + initial_forecast <= total_length:
        current_input_len = initial_input
        current_forecast_len = initial_forecast

        while (
            current_forecast_len >= min_forecast
            and start_index + current_input_len + current_forecast_len <= total_length
        ):
            input_slice = feature_matrix[start_index : start_index + current_input_len]
            target_slice = feature_matrix[
                start_index + current_input_len : start_index + current_input_len + current_forecast_len, 3
            ]

            X_list.append(input_slice)
            mask_list.append(np.ones((current_input_len, 1), dtype="float32"))
            y_list.append(target_slice)
            y_mask_list.append(np.ones((current_forecast_len,), dtype="float32"))
            group_id_list.append(group_id)

            # Reveal one more timestep: extend input and shorten remaining horizon while anchoring at start_index.
            current_input_len += 1
            current_forecast_len -= 1

        # Move to the next disjoint group anchored by the next unseen chunk.
        start_index += initial_input
        group_id += 1

    if not X_list:
        raise ValueError("No sliding windows could be generated with the given configuration.")

    X_padded = pad_sequences(
        X_list,
        maxlen=config.max_input_length,
        dtype="float32",
        padding="pre",
        value=0.0,
    )
    mask_padded = pad_sequences(
        mask_list,
        maxlen=config.max_input_length,
        dtype="float32",
        padding="pre",
        value=0.0,
    )
    y_padded = pad_sequences(
        y_list,
        maxlen=initial_forecast,
        dtype="float32",
        padding="post",
        value=0.0,
    )
    y_mask_padded = pad_sequences(
        y_mask_list,
        maxlen=initial_forecast,
        dtype="float32",
        padding="post",
        value=0.0,
    )
    group_ids = np.array(group_id_list, dtype=np.int32)

    return X_padded, y_padded, mask_padded, y_mask_padded, group_ids
