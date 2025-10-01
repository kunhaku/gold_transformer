"""Command-line entry point for running the training pipeline."""

from configs import DataConfig, ModelConfig
from pipelines import run_training_pipeline


def main() -> None:
    data_config = DataConfig()
    model_config = ModelConfig()
    run_training_pipeline(data_config, model_config)


if __name__ == "__main__":
    main()
