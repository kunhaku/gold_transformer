"""Pipeline entry points for orchestrating experiments."""

from .training_pipeline import run_training_pipeline

__all__ = ["run_training_pipeline"]
