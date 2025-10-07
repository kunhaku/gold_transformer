from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import shutil
import time
import uuid
from contextlib import contextmanager

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
def _train_step(model, optimizer, x, m, past_targets, y, y_mask):
    with tf.GradientTape() as tape:
        preds = model(x, mask=m, past_preds=past_targets, training=True)
        loss = masked_mse_loss(y, preds, y_mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return preds, loss


def _remove_path_if_exists(
    path: Path,
    *,
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
) -> None:
    for attempt in range(1, max_attempts + 1):
        if not path.exists():
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == max_attempts:
                raise
        except OSError as err:
            if attempt == max_attempts:
                raise err
        if not path.exists():
            return
        time.sleep(delay_seconds * attempt)
    if path.exists():
        raise PermissionError(f"Unable to remove existing path: {path}")


def _temporary_model_path(target: Path) -> Path:
    unique = uuid.uuid4().hex
    parent = target.parent or Path(".")
    if target.suffix:
        return parent / f"{target.stem}__tmp_save_{unique}{target.suffix}"
    name = target.name or "model"
    return parent / f"{name}__tmp_save_{unique}"


def _ensure_model_is_built(model: tf.keras.Model, dataset: SequenceDataset) -> None:
    if model.built:
        return
    x_sample = tf.convert_to_tensor(dataset.inputs[:1])
    mask_sample = tf.convert_to_tensor(dataset.input_mask[:1])
    past_sample = tf.convert_to_tensor(dataset.past_targets[:1])
    model(x_sample, mask=mask_sample, past_preds=past_sample, training=False)


def _rename_via_copy(src: str, dst: str, overwrite: bool) -> bool:
    """Best-effort fallback when atomic rename keeps failing on Windows."""

    try:
        src_path = Path(src)
        dst_path = Path(dst)
    except TypeError:
        return False

    if not src_path.exists():
        return False

    try:
        if dst_path.exists():
            if not overwrite:
                return False
            if dst_path.is_dir():
                shutil.rmtree(dst_path, ignore_errors=True)
            else:
                dst_path.unlink()
    except OSError:
        return False

    try:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
    except (OSError, shutil.Error):
        return False

    try:
        if src_path.is_dir():
            shutil.rmtree(src_path, ignore_errors=True)
        else:
            if src_path.exists():
                src_path.unlink()
    except OSError:
        pass

    return True


@contextmanager
def _gfile_rename_retry(
    *,
    initial_delay_seconds: float = 1.0,
    max_retry_time: float = 120.0,
) -> None:
    original_rename = tf.io.gfile.rename
    original_rename_v2 = getattr(tf.io.gfile, "rename_v2", None)

    try:
        from tensorflow.python.framework import errors_impl
    except ImportError:  # pragma: no cover - fallback for unexpected TF layouts
        errors_impl = None

    if errors_impl is None or original_rename is None:
        yield
        return

    def _rename_with_retry(src, dst, overwrite=False, *, _orig=original_rename):
        delay = initial_delay_seconds
        elapsed = 0.0
        while True:
            try:
                return _orig(src, dst, overwrite)
            except errors_impl.OpError as err:
                if "Failed to rename" not in str(err):
                    raise
                if elapsed >= max_retry_time:
                    if _rename_via_copy(src, dst, overwrite):
                        return
                    raise
                time.sleep(delay)
                elapsed += delay
                delay = min(delay * 1.5, 5.0)

    tf.io.gfile.rename = _rename_with_retry

    if original_rename_v2 is not None:

        def _rename_v2_with_retry(src, dst, overwrite=False, *, _orig=original_rename_v2):
            delay = initial_delay_seconds
            elapsed = 0.0
            while True:
                try:
                    return _orig(src, dst, overwrite)
                except errors_impl.OpError as err:
                    if "Failed to rename" not in str(err):
                        raise
                    if elapsed >= max_retry_time:
                        if _rename_via_copy(src, dst, overwrite):
                            return
                        raise
                    time.sleep(delay)
                    elapsed += delay
                    delay = min(delay * 1.5, 5.0)

        tf.io.gfile.rename_v2 = _rename_v2_with_retry

    try:
        yield
    finally:
        tf.io.gfile.rename = original_rename
        if original_rename_v2 is not None:
            tf.io.gfile.rename_v2 = original_rename_v2


def _save_model_with_retry(
    model: tf.keras.Model,
    target_path: Path,
    save_format: str,
    *,
    max_attempts: int = 3,
    delay_seconds: float = 1.5,
) -> None:
    last_error: Exception | None = None
    successful_temp_path: Path | None = None

    for attempt in range(1, max_attempts + 1):
        temp_path = _temporary_model_path(target_path)
        _remove_path_if_exists(temp_path)
        try:
            save_kwargs = {}
            if save_format and save_format.lower() == "tf":
                save_kwargs["options"] = tf.saved_model.SaveOptions()
            with _gfile_rename_retry(
                initial_delay_seconds=delay_seconds,
                max_retry_time=max(60.0, max_attempts * delay_seconds * 10),
            ):
                model.save(temp_path, save_format=save_format, **save_kwargs)
        except UnicodeDecodeError as err:
            last_error = err
            if attempt == max_attempts:
                break
            tqdm.write(
                f"Model save attempt {attempt} failed due to a transient filesystem encoding error; retrying..."
            )
            _remove_path_if_exists(temp_path, max_attempts=2, delay_seconds=delay_seconds)
            time.sleep(delay_seconds * attempt)
            continue
        except tf.errors.OpError as err:
            if "Failed to rename" not in str(err):
                _remove_path_if_exists(temp_path, max_attempts=2, delay_seconds=delay_seconds)
                raise
            last_error = err
            if attempt == max_attempts:
                break
            tqdm.write(
                f"Model save attempt {attempt} encountered a filesystem rename issue; retrying..."
            )
            _remove_path_if_exists(temp_path, max_attempts=2, delay_seconds=delay_seconds)
            time.sleep(delay_seconds * attempt)
            continue
        except PermissionError as err:
            last_error = err
            if attempt == max_attempts:
                break
            tqdm.write(
                f"Model save attempt {attempt} could not access the filesystem; retrying..."
            )
            _remove_path_if_exists(temp_path, max_attempts=2, delay_seconds=delay_seconds)
            time.sleep(delay_seconds * attempt)
            continue
        else:
            last_error = None
            successful_temp_path = temp_path
            break

    if last_error is not None:
        raise RuntimeError(f"Unable to save model to {target_path}") from last_error

    if successful_temp_path is None or not successful_temp_path.exists():
        raise RuntimeError(f"Model save succeeded but temporary artifact is missing at {target_path}")

    _remove_path_if_exists(target_path, max_attempts=5, delay_seconds=delay_seconds)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(successful_temp_path), str(target_path))


def train_model(
    train_data: SequenceDataset,
    config: ModelConfig,
    validation_data: SequenceDataset | None = None,
) -> Tuple[RecurrentTransformerModel, List[dict]]:
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

    train_size = train_data.inputs.shape[0]
    batch_size = min(config.batch_size, train_size)
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                train_data.inputs,
                train_data.input_mask,
                train_data.past_targets,
                train_data.targets,
                train_data.target_mask,
            )
        )
        .shuffle(train_size, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    log_path: Path | None = None
    summary_writer: tf.summary.SummaryWriter | None = None
    model_run_name: str | None = None
    if config.tensorboard_log_dir is not None:
        base_path = Path(config.tensorboard_log_dir)
        if not base_path.is_absolute():
            base_path = Path(config.model_dir) / base_path
        run_name = config.tensorboard_run_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_path = base_path / run_name
        model_run_name = run_name
        log_path.mkdir(parents=True, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(str(log_path))
    else:
        if config.tensorboard_run_name:
            model_run_name = config.tensorboard_run_name
        else:
            model_run_name = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    def _evaluate_dataset(dataset: SequenceDataset) -> Tuple[float, dict]:
        total_loss = 0.0
        steps = 0
        preds_buffer: List[np.ndarray] = []
        targets_buffer: List[np.ndarray] = []

        for group_id in np.unique(dataset.group_ids):
            indices = np.where(dataset.group_ids == group_id)[0]
            past_preds_eval = tf.zeros((1, forecast_length), dtype=tf.float32)

            for idx in indices:
                x_i = dataset.inputs[idx][None, ...]
                y_i = dataset.targets[idx][None, ...]
                m_i = dataset.input_mask[idx][None, ...]
                y_mask_i = dataset.target_mask[idx][None, ...]

                preds_eval = model(
                    x_i,
                    mask=m_i,
                    past_preds=past_preds_eval,
                    training=False,
                )
                loss_val = masked_mse_loss(y_i, preds_eval, y_mask_i)
                past_preds_eval = preds_eval

                preds_buffer.append(preds_eval.numpy().flatten())
                targets_buffer.append(y_i.flatten())

                total_loss += float(loss_val.numpy())
                steps += 1

        avg_loss = total_loss / steps if steps else 0.0
        metrics_dict = {}
        if preds_buffer:
            y_pred = np.vstack(preds_buffer)
            y_true = np.vstack(targets_buffer)
            metrics_dict = compute_regression_metrics(y_true, y_pred)

        return avg_loss, metrics_dict

    history: List[dict] = []

    use_early_stopping = validation_data is not None and config.early_stopping_patience > 0
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    for epoch in tqdm(range(config.epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        seen_examples = 0
        epoch_predictions: List[np.ndarray] = []
        epoch_targets: List[np.ndarray] = []

        for batch in train_ds:
            x_batch, m_batch, past_batch, y_batch, y_mask_batch = batch
            preds, loss_val = _train_step(
                model,
                optimizer,
                x_batch,
                m_batch,
                past_batch,
                y_batch,
                y_mask_batch,
            )

            epoch_predictions.append(preds.numpy())
            epoch_targets.append(y_batch.numpy())

            batch_size_val = y_batch.shape[0]
            if batch_size_val is None:
                batch_size_val = int(tf.shape(y_batch)[0])
            epoch_loss += float(loss_val.numpy()) * int(batch_size_val)
            seen_examples += int(batch_size_val)

        avg_loss = epoch_loss / seen_examples if seen_examples else 0.0
        metrics = {}
        if epoch_predictions:
            y_pred = np.vstack(epoch_predictions)
            y_true = np.vstack(epoch_targets)
            metrics = compute_regression_metrics(y_true, y_pred)

        val_loss = None
        val_metrics: dict[str, float] = {}
        if validation_data is not None:
            val_loss, val_metrics = _evaluate_dataset(validation_data)

        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar("loss/train", avg_loss, step=epoch + 1)
                for metric_name, value in metrics.items():
                    tf.summary.scalar(f"{metric_name}/train", value, step=epoch + 1)
                if val_loss is not None:
                    tf.summary.scalar("loss/val", val_loss, step=epoch + 1)
                    for metric_name, value in val_metrics.items():
                        tf.summary.scalar(f"{metric_name}/val", value, step=epoch + 1)
            summary_writer.flush()

        history_entry = {"epoch": epoch + 1, "loss": avg_loss, **metrics}
        if val_loss is not None:
            history_entry["val_loss"] = val_loss
            for metric_name, value in val_metrics.items():
                history_entry[f"val_{metric_name}"] = value

        history.append(history_entry)

        if use_early_stopping and val_loss is not None:
            if val_loss + config.early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    tqdm.write(
                        f"Early stopping triggered at epoch {epoch + 1}: val_loss={val_loss:.6f}"
                    )
                    break

    if use_early_stopping and best_weights is not None:
        model.set_weights(best_weights)

    if config.save_unique_subdir and model_run_name:
        model_path = config.model_path() / model_run_name
    else:
        model_path = config.model_path()
    config.last_run_model_path = model_path
    if config.save_format:
        ensure_directory(str(model_path))
        _ensure_model_is_built(model, train_data)
        _save_model_with_retry(model, model_path, config.save_format)

    if summary_writer is not None:
        summary_writer.close()

    return model, history
