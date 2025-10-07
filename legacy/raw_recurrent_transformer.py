"""Compatibility wrapper for training the recurrent transformer model."""

from __future__ import annotations

from typing import Optional

from configs import DataConfig, ModelConfig
from data.datasets import SequenceDataset, prepare_datasets
from models.train import train_model


def run_training(
    train_dataset: Optional[SequenceDataset] = None,
    model_config: Optional[ModelConfig] = None,
    data_config: Optional[DataConfig] = None,
):
    """Train the model using the refactored training utilities."""

    model_config = model_config or ModelConfig()
    if train_dataset is None:
        data_config = data_config or DataConfig()
        _, train_dataset, _, _, _ = prepare_datasets(data_config)

    model, history = train_model(train_dataset, model_config)
    return model


if __name__ == "__main__":
    run_training()
