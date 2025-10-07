from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from configs import DataConfig
from data.features import build_feature_frame
from data.ingest import load_mt5_data
from data.windowing import create_sliding_windows
from utils.io import ensure_directory


@dataclass(frozen=True)
class SequenceDataset:
    inputs: np.ndarray
    targets: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    past_targets: np.ndarray
    group_ids: np.ndarray

    def subset(self, mask: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(
            inputs=self.inputs[mask],
            targets=self.targets[mask],
            input_mask=self.input_mask[mask],
            target_mask=self.target_mask[mask],
            past_targets=self.past_targets[mask],
            group_ids=self.group_ids[mask],
        )

    def save(self, path: str) -> None:
        ensure_directory(path)
        np.savez_compressed(
            path,
            X_data=self.inputs,
            y_data=self.targets,
            mask_data=self.input_mask,
            y_mask_data=self.target_mask,
            past_targets=self.past_targets,
            group_ids=self.group_ids,
        )

    @staticmethod
    def load(path: str) -> "SequenceDataset":
        data = np.load(path)
        targets = data["y_data"]
        target_mask = data["y_mask_data"]
        group_ids = data["group_ids"]
        if "past_targets" in data:
            past_targets = data["past_targets"]
        else:
            past_targets = compute_past_targets(targets, target_mask, group_ids)
        return SequenceDataset(
            inputs=data["X_data"],
            targets=targets,
            input_mask=data["mask_data"],
            target_mask=target_mask,
            past_targets=past_targets,
            group_ids=group_ids,
        )


def compute_past_targets(
    targets: np.ndarray,
    target_mask: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Build a tensor containing the previous reveal step's targets per sample."""

    past_targets = np.zeros_like(targets, dtype=targets.dtype)
    unique_groups = np.unique(group_ids)
    zero_target = np.zeros(targets.shape[1], dtype=targets.dtype)
    zero_mask = np.zeros(target_mask.shape[1], dtype=target_mask.dtype)

    for group_id in unique_groups:
        indices = np.where(group_ids == group_id)[0]
        prev_target = zero_target
        prev_mask = zero_mask
        for idx in indices:
            past_targets[idx] = prev_target * prev_mask.astype(targets.dtype)
            prev_target = targets[idx]
            prev_mask = target_mask[idx]

    return past_targets


def build_sequence_dataset(
    config: DataConfig,
) -> Tuple[
    SequenceDataset,
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, float]],
]:
    """Build the full dataset from raw data to padded windows."""

    raw_df = load_mt5_data(config)
    feature_df = build_feature_frame(raw_df)

    feature_matrix = feature_df[config.feature_columns()].values
    X, y, mask, y_mask, group_ids = create_sliding_windows(feature_matrix, config)
    dataset = SequenceDataset(
        inputs=X,
        targets=y,
        input_mask=mask,
        target_mask=y_mask,
        past_targets=np.zeros_like(y, dtype=y.dtype),
        group_ids=group_ids,
    )
    return _scale_dataset_per_group(dataset, config)


def split_train_test(dataset: SequenceDataset, config: DataConfig) -> Tuple[SequenceDataset, SequenceDataset]:
    unique_groups = np.unique(dataset.group_ids)
    num_groups = len(unique_groups)
    if num_groups == 0:
        raise ValueError("Dataset contains no groups to split.")

    cut_idx = max(1, int(num_groups * config.train_ratio))
    train_groups = unique_groups[:cut_idx]
    train_mask = np.isin(dataset.group_ids, train_groups)
    test_mask = ~train_mask

    return dataset.subset(train_mask), dataset.subset(test_mask)


def prepare_datasets(
    config: DataConfig,
) -> Tuple[
    SequenceDataset,
    SequenceDataset,
    SequenceDataset,
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, float]],
]:
    """Convenience helper returning the full, train, and test datasets."""

    (
        full_dataset,
        scaler_metadata,
        target_metadata,
    ) = build_sequence_dataset(config)
    train_dataset, test_dataset = split_train_test(full_dataset, config)
    return full_dataset, train_dataset, test_dataset, scaler_metadata, target_metadata


def _scale_dataset_per_group(
    dataset: SequenceDataset,
    config: DataConfig,
) -> Tuple[
    SequenceDataset,
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, float]],
]:
    feature_names = config.feature_columns()
    column_index = {name: idx for idx, name in enumerate(feature_names)}

    scale_columns = [
        col
        for col in ["open", "high", "low", "close", "boll_upper", "boll_lower", "MA3", "MA12"]
        if col in column_index
    ]

    inputs = np.array(dataset.inputs, copy=True)
    targets = np.array(dataset.targets, copy=True)
    input_mask = dataset.input_mask
    target_mask = dataset.target_mask
    group_ids = dataset.group_ids

    feature_metadata: Dict[str, Dict[str, Dict[str, float]]] = {}
    target_metadata: Dict[str, Dict[str, float]] = {}

    unique_groups = np.unique(group_ids)
    for group_id in unique_groups:
        group_key = str(int(group_id))
        group_indices = np.where(group_ids == group_id)[0]

        group_feature_meta: Dict[str, Dict[str, float]] = {}
        close_stats: Dict[str, float] | None = None

        for column in scale_columns:
            col_idx = column_index[column]
            values = []
            for sample_idx in group_indices:
                valid = input_mask[sample_idx, :, 0] > 0.5
                if np.any(valid):
                    values.append(inputs[sample_idx, valid, col_idx])
            if values:
                concatenated = np.concatenate(values)
                mean = float(concatenated.mean())
                scale = float(concatenated.std())
                if scale == 0.0:
                    scale = 1.0
            else:
                mean = 0.0
                scale = 1.0

            for sample_idx in group_indices:
                valid = input_mask[sample_idx, :, 0] > 0.5
                if np.any(valid):
                    inputs[sample_idx, valid, col_idx] = (
                        inputs[sample_idx, valid, col_idx] - mean
                    ) / scale

            group_feature_meta[column] = {"mean": mean, "scale": scale}
            if column == "close":
                close_stats = {"mean": mean, "scale": scale}

        feature_metadata[group_key] = group_feature_meta

        if close_stats is not None:
            mean = close_stats["mean"]
            scale = close_stats["scale"]
        else:
            target_values = []
            for sample_idx in group_indices:
                valid = target_mask[sample_idx] > 0.5
                if np.any(valid):
                    target_values.append(targets[sample_idx, valid])
            if target_values:
                concatenated = np.concatenate(target_values)
                mean = float(concatenated.mean())
                scale = float(concatenated.std())
                if scale == 0.0:
                    scale = 1.0
            else:
                mean = 0.0
                scale = 1.0

        for sample_idx in group_indices:
            valid = target_mask[sample_idx] > 0.5
            if np.any(valid):
                targets[sample_idx, valid] = (targets[sample_idx, valid] - mean) / scale

        target_metadata[group_key] = {"mean": mean, "scale": scale}

    past_targets = compute_past_targets(targets, target_mask, group_ids)
    scaled_dataset = SequenceDataset(
        inputs=inputs,
        targets=targets,
        input_mask=input_mask,
        target_mask=target_mask,
        past_targets=past_targets,
        group_ids=group_ids,
    )
    return scaled_dataset, feature_metadata, target_metadata
