from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Configuration for data ingestion and preprocessing."""

    db_path: str = "mt5_data.db"
    table_name: str = "main.XAUUSD"
    initial_input_length: int = 36
    max_input_length: int = 48
    initial_forecast_length: int = 24
    min_forecast_length: int = 10
    train_ratio: float = 0.8
    num_samples_to_visualize: int = 5
    artifact_dir: str = "artifacts"
    train_dataset_filename: str = "train_dataset.npz"
    test_dataset_filename: str = "test_dataset.npz"
    scaler_filename: str = "scaler_params.npz"
    prediction_db_filename: str = "predictions.db"
    base_features: List[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
        ]
    )
    additional_indicators: List[str] = field(
        default_factory=lambda: [
            "oc_dist",
            "oh_dist",
            "hl_dist",
            "lc_dist",
            "RSI",
            "MA3",
            "MA12",
            "MA_diff",
            "boll_upper",
            "boll_lower",
            "boll_bandwidth",
        ]
    )

    def feature_columns(self) -> List[str]:
        """Return the ordered list of feature columns used for modeling."""

        return self.base_features + self.additional_indicators

    def artifact_path(self, filename: str) -> Path:
        """Return an absolute path inside the artifact directory."""

        return Path(self.artifact_dir) / filename
