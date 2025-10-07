from __future__ import annotations

from typing import Tuple

from configs import DataConfig, ModelConfig
from data.datasets import SequenceDataset, prepare_datasets
from data.scaling import save_scaler_metadata
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


def run_training_pipeline(
    data_config: DataConfig | None = None, model_config: ModelConfig | None = None
) -> dict:
    data_config = data_config or DataConfig()
    model_config = model_config or ModelConfig()

    ensure_directory(str(data_config.artifact_path("placeholder")))
    ensure_directory(str(model_config.model_path()))

    (
        full_dataset,
        train_dataset,
        test_dataset,
        feature_scaler_metadata,
        target_scaler_metadata,
    ) = prepare_datasets(data_config)
    train_path, test_path = _save_artifacts(data_config, train_dataset, test_dataset)

    model, history = train_model(train_dataset, model_config, validation_data=test_dataset)
    model_artifact_path = (
        model_config.last_run_model_path if model_config.last_run_model_path is not None else model_config.model_path()
    )

    prediction_db_path = str(data_config.artifact_path(data_config.prediction_db_filename))
    scaler_metadata_path = str(data_config.artifact_path(data_config.scaler_metadata_filename))
    save_scaler_metadata(
        scaler_metadata_path,
        feature_metadata=feature_scaler_metadata,
        target_metadata=target_scaler_metadata if target_scaler_metadata else None,
    )
    inference_metrics = run_inference(
        model,
        test_dataset,
        prediction_db_path,
        scaler_metadata_path=scaler_metadata_path,
    )

    run_visual_tool(
        {
            "db_path": prediction_db_path,
            "dataset_path": test_path,
            "forecast_length": model_config.forecast_length or test_dataset.targets.shape[1],
            "scaler_metadata_path": scaler_metadata_path,
        }
    )

    return {
        "artifacts": {
            "train_dataset": train_path,
            "test_dataset": test_path,
            "prediction_db": prediction_db_path,
            "scaler_metadata": scaler_metadata_path,
            "model_path": str(model_artifact_path),
        },
        "training_history": history,
        "inference_metrics": inference_metrics,
        "model": model,
        "scaler_metadata": {
            "features": feature_scaler_metadata,
            "target": target_scaler_metadata,
        },
        "datasets": {
            "full": full_dataset,
            "train": train_dataset,
            "test": test_dataset,
        },
    }


if __name__ == "__main__":
    run_training_pipeline()
