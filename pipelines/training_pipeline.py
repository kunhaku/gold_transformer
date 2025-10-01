from __future__ import annotations

from typing import Tuple

from configs import DataConfig, ModelConfig
from data.datasets import SequenceDataset, prepare_datasets
from evaluation import run_inference
from models import train_model
from utils.io import ensure_directory
from visual_tool import run_visual_tool


def _save_artifacts(data_config: DataConfig, train: SequenceDataset, test: SequenceDataset) -> Tuple[str, str]:
    train_path = str(data_config.artifact_path(data_config.train_dataset_filename))
    test_path = str(data_config.artifact_path(data_config.test_dataset_filename))
    train.save(train_path)
    test.save(test_path)
    return train_path, test_path


def run_training_pipeline(data_config: DataConfig | None = None, model_config: ModelConfig | None = None) -> dict:
    data_config = data_config or DataConfig()
    model_config = model_config or ModelConfig()

    ensure_directory(str(data_config.artifact_path("placeholder")))
    ensure_directory(str(model_config.model_path()))

    _, train_dataset, test_dataset = prepare_datasets(data_config)
    train_path, test_path = _save_artifacts(data_config, train_dataset, test_dataset)

    model, history = train_model(train_dataset, model_config)

    prediction_db_path = str(data_config.artifact_path(data_config.prediction_db_filename))
    inference_metrics = run_inference(model, test_dataset, prediction_db_path)

    run_visual_tool(
        {
            "db_path": prediction_db_path,
            "dataset_path": test_path,
            "forecast_length": model_config.forecast_length or test_dataset.targets.shape[1],
        }
    )

    return {
        "artifacts": {
            "train_dataset": train_path,
            "test_dataset": test_path,
            "prediction_db": prediction_db_path,
            "model_path": str(model_config.model_path()),
        },
        "training_history": history,
        "inference_metrics": inference_metrics,
    }


if __name__ == "__main__":
    run_training_pipeline()
