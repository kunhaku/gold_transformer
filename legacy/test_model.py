"""Compatibility wrapper for running inference on the trained model."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from configs import DataConfig, ModelConfig
from data.datasets import SequenceDataset
from evaluation.inference import run_inference


def run_test_model(
    model: Optional[tf.keras.Model] = None,
    test_dataset: Optional[SequenceDataset] = None,
    data_config: Optional[DataConfig] = None,
    model_config: Optional[ModelConfig] = None,
    model_path: Optional[str] = None,
):
    model_config = model_config or ModelConfig()
    data_config = data_config or DataConfig()

    if test_dataset is None:
        test_dataset = SequenceDataset.load(
            str(data_config.artifact_path(data_config.test_dataset_filename))
        )

    if model is None:
        path = model_path or str(model_config.model_path())
        model = tf.keras.models.load_model(path, compile=False)

    prediction_db_path = str(data_config.artifact_path(data_config.prediction_db_filename))
    return run_inference(model, test_dataset, prediction_db_path)


if __name__ == "__main__":
    run_test_model()
