"""
Automated Comprehensive Test Suite for Insurance-Cost-AI
Validates dataset, preprocessor, model inference, and page execution.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction_helper import get_predictor
from src.analytics.analytics_helper import (
    compute_kpis,
    create_cost_distribution_chart,
    create_age_vs_cost_scatter,
    create_risk_gauge_chart
)

def test_dataset_exists():
    """Verify raw and processed datasets exist and have >= 5000 records."""
    raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'insurance_dataset.csv')
    proc_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')

    assert os.path.exists(raw_path), "Raw dataset file missing"
    assert os.path.exists(proc_path), "Processed dataset file missing"

    df_raw = pd.read_csv(raw_path)
    df_proc = pd.read_csv(proc_path)

    assert len(df_raw) >= 5000, f"Expected >= 5000 raw records, got {len(df_raw)}"
    assert len(df_proc) >= 5000, f"Expected >= 5000 processed records, got {len(df_proc)}"

    # Check required columns
    required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    for c in required_cols:
        assert c in df_raw.columns, f"Missing required column {c} in raw data"
        assert c in df_proc.columns, f"Missing required column {c} in processed data"

    # Check engineered columns
    assert 'bmi_category' in df_proc.columns
    assert 'age_group' in df_proc.columns
    assert 'risk_score' in df_proc.columns
    assert 'risk_category' in df_proc.columns
    print(f"[OK] Dataset verified: {len(df_raw)} records, all required & engineered columns present.")

def test_model_loading_and_inference():
    """Verify saved models load and generate valid predictions."""
    predictor = get_predictor()
    assert predictor.is_loaded, "Predictor failed to load model and preprocessor artifacts"

    # Test young non-smoker
    pred1 = predictor.predict_single(age=25, sex='female', bmi=21.5, children=0, smoker='no', region='southwest')
    assert pred1['predicted_annual_cost'] > 0
    assert pred1['predicted_monthly_cost'] == round(pred1['predicted_annual_cost'] / 12, 2)
    assert pred1['risk_category'] == 'LOW'
    assert len(pred1['explanation_insights']) > 0

    # Test senior obese smoker
    pred2 = predictor.predict_single(age=60, sex='male', bmi=36.0, children=2, smoker='yes', region='southeast')
    assert pred2['predicted_annual_cost'] > pred1['predicted_annual_cost']
    assert pred2['risk_category'] == 'HIGH'
    assert pred2['risk_score'] > 70

    print(f"[OK] Model predictions verified:")
    print(f"  Young healthy non-smoker: ${pred1['predicted_annual_cost']:,.2f} ({pred1['risk_category']})")
    print(f"  Senior obese smoker:      ${pred2['predicted_annual_cost']:,.2f} ({pred2['risk_category']})")

def test_analytics_and_charts():
    """Verify analytics helper and chart generation functions."""
    proc_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')
    df = pd.read_csv(proc_path)

    kpis = compute_kpis(df)
    assert kpis['total_customers'] >= 5000
    assert kpis['avg_cost'] > 0
    assert 0 <= kpis['high_risk_pct'] <= 100

    fig1 = create_cost_distribution_chart(df)
    assert fig1 is not None

    fig2 = create_age_vs_cost_scatter(df)
    assert fig2 is not None

    fig3 = create_risk_gauge_chart(45.0, 'MEDIUM')
    assert fig3 is not None

    print("[OK] Analytics and Plotly chart generators verified successfully.")

if __name__ == '__main__':
    test_dataset_exists()
    test_model_loading_and_inference()
    test_analytics_and_charts()
    print("\n[SUCCESS] All automated validation tests passed successfully!")
