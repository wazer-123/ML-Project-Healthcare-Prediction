"""
Model Evaluation Utilities
Computes actuarial performance metrics (MAE, MSE, RMSE, R²) and extracts
diagnostic analytics for model evaluation and reporting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE, and R2 score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }

def calculate_residuals(y_true, y_pred) -> np.ndarray:
    """Calculate residuals: Actual - Predicted."""
    return np.asarray(y_true) - np.asarray(y_pred)
