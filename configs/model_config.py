from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    num_layers: int = 4
    embed_dim: int = 64
    num_heads: int = 8
    ff_dim: int = 64
    dropout_rate: float = 0.1
    forecast_length: int | None = None
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 64
    model_dir: str = "models"
    model_name: str = "recurrent_transformer"
    save_format: str = "tf"
    tensorboard_log_dir: str | None = "logs"
    tensorboard_run_name: str | None = None
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    save_unique_subdir: bool = True
    last_run_model_path: Path | None = field(default=None, init=False, repr=False)

    def model_path(self) -> Path:
        """Return the path where the trained model should be saved."""

        return Path(self.model_dir) / self.model_name
