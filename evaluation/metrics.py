from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R2 scores for the given predictions."""

    mae_val = mean_absolute_error(y_true, y_pred)
    mse_val = mean_squared_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mse_val))
    r2_val = r2_score(y_true, y_pred)
    return {"mae": mae_val, "rmse": rmse_val, "r2": r2_val}
