"""
Phase 3 Machine Learning Validation Test Suite
Performs automated checks on all model artifacts, feature matrices,
input data types, regression metrics, and diverse applicant profiles.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.evaluate_models import compute_regression_metrics
from src.prediction.prediction_helper import get_predictor
from src.data_processing.preprocess import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL

def test_ml_validation():
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    
    # 1. Load artifacts
    best_model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessing_pipeline.pkl'))
    all_models = joblib.load(os.path.join(models_dir, 'all_models.pkl'))

    with open(os.path.join(models_dir, 'model_evaluation_report.json')) as fp:
        report = json.load(fp)

    # 2. Test split validation
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv'))
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_test_trans = preprocessor.transform(X_test)

    print("=== MODEL BENCHMARK METRIC VERIFICATION ===")
    for name, model in all_models.items():
        preds = model.predict(X_test_trans)
        m = compute_regression_metrics(y_test, preds)
        rep_m = report['models'][name]['metrics']
        
        # Verify exact reproducibility
        assert abs(m['R2'] - rep_m['R2']) < 1e-4, f"R2 mismatch for {name}"
        assert abs(m['RMSE'] - rep_m['RMSE']) < 1e-2, f"RMSE mismatch for {name}"
        assert abs(m['MAE'] - rep_m['MAE']) < 1e-2, f"MAE mismatch for {name}"
        print(f"[OK] {name:<30} | R²: {m['R2']:.4f} | RMSE: ${m['RMSE']:,.2f} | MAE: ${m['MAE']:,.2f}")

    # 3. Predictor profiles
    print("\n=== MULTI-PROFILE PREDICTION INFERENCE TEST ===")
    p = get_predictor()
    test_profiles = [
        ('Young Healthy Non-Smoker', 21, 'female', 21.5, 0, 'no', 'southwest'),
        ('Mid-Age Normal Non-Smoker', 40, 'male', 24.2, 1, 'no', 'northwest'),
        ('Mid-Age Overweight Non-Smoker', 45, 'male', 28.5, 2, 'no', 'southeast'),
        ('Mid-Age Obese Non-Smoker', 45, 'female', 34.0, 2, 'no', 'northeast'),
        ('Young Smoker Normal Weight', 25, 'male', 23.0, 0, 'yes', 'southwest'),
        ('Senior Obese Smoker (Highest Risk)', 62, 'male', 38.0, 3, 'yes', 'southeast'),
        ('Boundary Min (18yo, BMI 15.0)', 18, 'female', 15.0, 0, 'no', 'northeast'),
        ('Boundary Max (65yo, BMI 50.0, 5 kids)', 65, 'female', 50.0, 5, 'yes', 'southeast')
    ]

    for desc, age, sex, bmi, children, smoker, region in test_profiles:
        res = p.predict_single(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)
        assert res['predicted_annual_cost'] > 0
        assert res['predicted_monthly_cost'] == round(res['predicted_annual_cost'] / 12.0, 2)
        assert 0 <= res['risk_score'] <= 100
        assert res['risk_category'] in ['LOW', 'MEDIUM', 'HIGH']
        print(f"[OK] {desc:<38} -> ${res['predicted_annual_cost']:>10,.2f}/yr (${res['predicted_monthly_cost']:>8,.2f}/mo) | Score: {res['risk_score']:>4.1f} ({res['risk_category']})")

    print("\n[SUCCESS] Phase 3 ML Model Validation passed completely!")

if __name__ == '__main__':
    test_ml_validation()
