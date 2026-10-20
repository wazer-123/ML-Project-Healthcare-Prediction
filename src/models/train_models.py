"""
Model Training & Selection Pipeline
Trains Linear Regression, Random Forest, and Gradient Boosting models,
evaluates performance metrics, automatically selects the best model,
and persists artifacts via Joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.generate_dataset import save_raw_dataset
from src.data_processing.preprocess import run_data_pipeline, build_preprocessing_pipeline, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL
from src.models.evaluate_models import compute_regression_metrics, calculate_residuals

def train_and_evaluate_all():
    """Execute training pipeline for all candidate models and save winning model."""
    raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'insurance_dataset.csv')
    processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Ensure dataset is generated and preprocessed
    if not os.path.exists(raw_path):
        print("[Pipeline] Raw data missing. Generating dataset...")
        save_raw_dataset(raw_path, n_samples=6000)

    if not os.path.exists(processed_path):
        print("[Pipeline] Processed data missing. Preprocessing...")
        run_data_pipeline(raw_path, processed_path)

    df = pd.read_csv(processed_path)
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 80/20 Train/Test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Fit preprocessor
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract feature names
    feature_names = [f.replace('num__', '').replace('cat__', '') for f in preprocessor.get_feature_names_out()]

    # Candidate models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
    }

    results = {}
    fitted_models = {}
    best_name = None
    best_r2 = -float('inf')

    # Test sample subsample for fast dashboard charting (first 500 test points)
    sample_size = min(500, len(y_test))
    y_test_sample = y_test.iloc[:sample_size].tolist()

    for name, model in candidate_models.items():
        print(f"[Training] Fitting {name}...")
        model.fit(X_train_transformed, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test_transformed)
        metrics = compute_regression_metrics(y_test, y_pred)
        residuals = calculate_residuals(y_test, y_pred)

        # Feature importance extraction
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).tolist()
            # Normalize to sum 1
            importances = (importances / np.sum(importances)).tolist()
        else:
            importances = [1.0 / len(feature_names)] * len(feature_names)

        results[name] = {
            'metrics': metrics,
            'feature_importances': dict(zip(feature_names, [round(float(v), 5) for v in importances])),
            'y_pred_sample': [round(float(p), 2) for p in y_pred[:sample_size]],
            'residuals_sample': [round(float(r), 2) for r in residuals[:sample_size]]
        }

        print(f"  -> {name} | R2: {metrics['R2']} | RMSE: ${metrics['RMSE']} | MAE: ${metrics['MAE']}")

        # Model selection criterion: Highest R2 score
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_name = name

    print(f"\n[Selection] Winning Model: '{best_name}' with R² = {best_r2}")

    # Persist best model and preprocessor
    best_model = fitted_models[best_name]
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.pkl')
    all_models_path = os.path.join(models_dir, 'all_models.pkl')

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, pipeline_path)
    joblib.dump(fitted_models, all_models_path)

    # Save comprehensive report for the dashboard
    report = {
        'best_model_name': best_name,
        'selection_criterion': 'Highest Test R² and Lowest Test RMSE',
        'feature_names': feature_names,
        'y_true_sample': [round(float(y), 2) for y in y_test_sample],
        'models': results,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    report_path = os.path.join(models_dir, 'model_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[Export] Saved best model to: {best_model_path}")
    print(f"[Export] Saved preprocessor to: {pipeline_path}")
    print(f"[Export] Saved evaluation report to: {report_path}")

    return best_name, results

if __name__ == '__main__':
    train_and_evaluate_all()
