"""
evaluation.py - Neural Network Regression Performance Metrics.

Calculates standard regression evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
"""

import numpy as np

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error: mean(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error: sqrt(mean((y_true - y_pred)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Coefficient of Determination (R²):
    R² = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all regression metrics and return as a clean summary dict.
    """
    y_t = np.array(y_true, dtype=np.float32).flatten()
    y_p = np.array(y_pred, dtype=np.float32).flatten()

    mae = compute_mae(y_t, y_p)
    rmse = compute_rmse(y_t, y_p)
    r2 = compute_r2(y_t, y_p)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
