"""Backwards-compatible accessors for configuration dataclasses."""

from configs import DataConfig, ModelConfig


data_config = DataConfig()
model_config = ModelConfig()

# Legacy dictionary-style access for existing scripts.
config = {
    "data": data_config,
    "model": model_config,
}

__all__ = ["data_config", "model_config", "config"]
