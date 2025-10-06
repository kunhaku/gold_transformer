from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configs import ModelConfig
from data.datasets import SequenceDataset
from evaluation.metrics import compute_regression_metrics
from models.losses import masked_mse_loss
from models.transformer import RecurrentTransformerModel
from utils.io import ensure_directory


@tf.function
def _train_step(model, optimizer, x, m, y, y_mask, past_preds):
    with tf.GradientTape() as tape:
        preds = model(x, mask=m, past_preds=past_preds, training=True)
        loss = masked_mse_loss(y, preds, y_mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return preds, loss


def train_model(train_data: SequenceDataset, config: ModelConfig) -> Tuple[RecurrentTransformerModel, List[dict]]:
    """Train the recurrent transformer model on the provided dataset."""

    input_dim = train_data.inputs.shape[2]
    forecast_length = train_data.targets.shape[1]
    if config.forecast_length is None:
        config.forecast_length = forecast_length
    elif config.forecast_length != forecast_length:
        raise ValueError(
            f"Configured forecast length ({config.forecast_length}) does not match dataset ({forecast_length})."
        )

    model = RecurrentTransformerModel(
        input_dim=input_dim,
        forecast_length=forecast_length,
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    history: List[dict] = []
    unique_groups = np.unique(train_data.group_ids)
    for epoch in tqdm(range(config.epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        steps = 0
        epoch_predictions: List[np.ndarray] = []
        epoch_targets: List[np.ndarray] = []

        for group_id in unique_groups:
            indices = np.where(train_data.group_ids == group_id)[0]
            past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

            for idx in indices:
                x_i = train_data.inputs[idx][None, ...]
                y_i = train_data.targets[idx][None, ...]
                m_i = train_data.input_mask[idx][None, ...]
                y_mask_i = train_data.target_mask[idx][None, ...]

                preds, loss_val = _train_step(
                    model,
                    optimizer,
                    x_i,
                    m_i,
                    y_i,
                    y_mask_i,
                    past_preds,
                )
                past_preds = preds

                epoch_predictions.append(preds.numpy().flatten())
                epoch_targets.append(y_i.flatten())

                epoch_loss += float(loss_val.numpy())
                steps += 1

        avg_loss = epoch_loss / steps if steps else 0.0
        metrics = {}
        if epoch_predictions:
            y_pred = np.vstack(epoch_predictions)
            y_true = np.vstack(epoch_targets)
            metrics = compute_regression_metrics(y_true, y_pred)

        history.append({"epoch": epoch + 1, "loss": avg_loss, **metrics})

    model_path = config.model_path()
    if config.save_format:
        ensure_directory(str(model_path))
        model.save(model_path, save_format=config.save_format)

    return model, history
