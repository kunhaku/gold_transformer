from __future__ import annotations

import tensorflow as tf


def masked_mse_loss(y_true, y_pred, y_mask):
    """Compute the mean squared error only on unmasked positions."""

    sq_err = tf.square(y_true - y_pred)
    masked_sq_err = sq_err * y_mask
    sum_err = tf.reduce_sum(masked_sq_err, axis=1)
    valid_counts = tf.reduce_sum(y_mask, axis=1)
    valid_counts = tf.where(valid_counts == 0, 1.0, valid_counts)
    mse_per_sample = sum_err / valid_counts
    return tf.reduce_mean(mse_per_sample)
