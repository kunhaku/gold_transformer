from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
    group_ids: np.ndarray

    def subset(self, mask: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(
            inputs=self.inputs[mask],
            targets=self.targets[mask],
            input_mask=self.input_mask[mask],
            target_mask=self.target_mask[mask],
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
            group_ids=self.group_ids,
        )

    @staticmethod
    def load(path: str) -> "SequenceDataset":
        data = np.load(path)
        return SequenceDataset(
            inputs=data["X_data"],
            targets=data["y_data"],
            input_mask=data["mask_data"],
            target_mask=data["y_mask_data"],
            group_ids=data["group_ids"],
        )


def build_sequence_dataset(config: DataConfig) -> SequenceDataset:
    """Build the full dataset from raw data to padded windows."""

    raw_df = load_mt5_data(config)
    feature_df = build_feature_frame(raw_df)
    feature_matrix = feature_df[config.feature_columns()].values
    X, y, mask, y_mask, group_ids = create_sliding_windows(feature_matrix, config)
    return SequenceDataset(X, y, mask, y_mask, group_ids)


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


def prepare_datasets(config: DataConfig) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Convenience helper returning the full, train, and test datasets."""

    full_dataset = build_sequence_dataset(config)
    train_dataset, test_dataset = split_train_test(full_dataset, config)
    return full_dataset, train_dataset, test_dataset
