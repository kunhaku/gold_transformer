from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from data.datasets import SequenceDataset

ScalerMetadata = Dict[str, np.ndarray]


def fit_sequence_scaler(dataset: "SequenceDataset") -> ScalerMetadata:
    """Compute feature-wise scaling statistics from valid timesteps."""

    if dataset.inputs.size == 0:
        raise ValueError("Cannot fit scaler on an empty dataset.")

    input_mask = dataset.input_mask.astype(bool)
    if input_mask.shape != dataset.inputs.shape:
        input_mask = np.broadcast_to(input_mask, dataset.inputs.shape)

    masked_inputs = np.ma.array(dataset.inputs, mask=~input_mask)
    input_mean = np.asarray(masked_inputs.mean(axis=(0, 1)).filled(0.0))
    input_std = np.asarray(masked_inputs.std(axis=(0, 1)).filled(1.0))
    input_std = np.where(input_std > 0, input_std, 1.0)

    target_mask = dataset.target_mask.astype(bool)
    masked_targets = np.ma.array(dataset.targets, mask=~target_mask)
    target_mean = np.asarray(masked_targets.mean(axis=0).filled(0.0))
    target_std = np.asarray(masked_targets.std(axis=0).filled(1.0))
    target_std = np.where(target_std > 0, target_std, 1.0)

    dtype = dataset.inputs.dtype
    target_dtype = dataset.targets.dtype

    scaler: ScalerMetadata = {
        "input_mean": input_mean.astype(dtype, copy=False),
        "input_std": input_std.astype(dtype, copy=False),
        "target_mean": target_mean.astype(target_dtype, copy=False),
        "target_std": target_std.astype(target_dtype, copy=False),
    }

    return scaler


def apply_sequence_scaler(
    dataset: "SequenceDataset", scaler: ScalerMetadata
) -> Tuple["SequenceDataset", ScalerMetadata]:
    """Scale dataset inputs and targets in-place using the provided metadata."""

    input_mean = scaler["input_mean"].astype(dataset.inputs.dtype, copy=False).reshape(1, 1, -1)
    input_std = scaler["input_std"].astype(dataset.inputs.dtype, copy=False).reshape(1, 1, -1)

    target_mean = scaler["target_mean"].astype(dataset.targets.dtype, copy=False).reshape(1, -1)
    target_std = scaler["target_std"].astype(dataset.targets.dtype, copy=False).reshape(1, -1)

    input_mask = dataset.input_mask.astype(bool)
    if input_mask.shape[-1] == 1 and dataset.inputs.ndim == 3:
        input_mask = np.broadcast_to(input_mask, dataset.inputs.shape)

    dataset.inputs[...] = np.where(
        input_mask,
        (dataset.inputs - input_mean) / input_std,
        dataset.inputs,
    )

    target_mask = dataset.target_mask.astype(bool)
    dataset.targets[...] = np.where(
        target_mask,
        (dataset.targets - target_mean) / target_std,
        dataset.targets,
    )

    return replace(dataset, scaler=scaler), scaler
