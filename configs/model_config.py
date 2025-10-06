from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    num_layers: int = 2
    embed_dim: int = 16
    num_heads: int = 4
    ff_dim: int = 64
    dropout_rate: float = 0.1
    forecast_length: int | None = None
    learning_rate: float = 1e-3
    epochs: int = 10
    model_dir: str = "models"
    model_name: str = "recurrent_transformer"
    save_format: str = "tf"

    def model_path(self) -> Path:
        """Return the path where the trained model should be saved."""

        return Path(self.model_dir) / self.model_name
