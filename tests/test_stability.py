"""
Deep Stability & Quality Audit Suite for Insurance-Cost-AI
Tests comprehensive input permutations, extreme boundary values,
dataframe schema consistency, scenario matrices, and export outputs.
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
    create_cost_by_age_group_chart,
    create_cost_by_bmi_category_chart,
    create_cost_by_smoker_chart,
    create_cost_by_gender_chart,
    create_cost_by_region_chart,
    create_children_vs_cost_chart,
    create_age_vs_cost_scatter,
    create_bmi_vs_cost_scatter,
    create_risk_distribution_chart,
    create_risk_gauge_chart,
    create_actual_vs_predicted_chart,
    create_residual_plot,
    create_error_distribution_chart,
    create_feature_importance_chart,
    create_scenario_comparison_chart
)
from pages.model_performance import load_evaluation_report

def test_inference_boundary_permutations():
    """Test predictor with boundary values and combinations."""
    predictor = get_predictor()
    assert predictor.is_loaded, "Predictor must be loaded"

    test_cases = [
        # Min boundary
        {'age': 18, 'sex': 'female', 'bmi': 15.0, 'children': 0, 'smoker': 'no', 'region': 'southwest'},
        # Max boundary
        {'age': 85, 'sex': 'male', 'bmi': 55.0, 'children': 5, 'smoker': 'yes', 'region': 'southeast'},
        # Intermediate combinations
        {'age': 25, 'sex': 'male', 'bmi': 22.4, 'children': 1, 'smoker': 'no', 'region': 'northwest'},
        {'age': 45, 'sex': 'female', 'bmi': 31.0, 'children': 3, 'smoker': 'yes', 'region': 'northeast'},
        {'age': 60, 'sex': 'male', 'bmi': 28.5, 'children': 0, 'smoker': 'no', 'region': 'southwest'},
        {'age': 35, 'sex': 'female', 'bmi': 40.2, 'children': 2, 'smoker': 'no', 'region': 'southeast'}
    ]

    for i, tc in enumerate(test_cases):
        res = predictor.predict_single(
            age=tc['age'],
            sex=tc['sex'],
            bmi=tc['bmi'],
            children=tc['children'],
            smoker=tc['smoker'],
            region=tc['region']
        )
        assert res['predicted_annual_cost'] > 0, f"Case {i}: Predicted annual cost must be > 0"
        assert res['predicted_monthly_cost'] > 0, f"Case {i}: Predicted monthly cost must be > 0"
        assert 0 <= res['risk_score'] <= 100, f"Case {i}: Risk score must be between 0 and 100"
        assert res['risk_category'] in ['LOW', 'MEDIUM', 'HIGH'], f"Case {i}: Invalid risk category"
        assert len(res['explanation_insights']) > 0, f"Case {i}: Explanation insights must be present"

    print(f"[OK] Verified {len(test_cases)} inference boundary test cases successfully.")

def test_all_chart_generators():
    """Verify all 16 Plotly chart generators execute with zero exceptions."""
    proc_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')
    df = pd.read_csv(proc_path)

    # Demographic & Cost charts
    c1 = create_cost_distribution_chart(df)
    c2 = create_cost_by_age_group_chart(df)
    c3 = create_cost_by_bmi_category_chart(df)
    c4 = create_cost_by_smoker_chart(df)
    c5 = create_cost_by_gender_chart(df)
    c6 = create_cost_by_region_chart(df)
    c7 = create_children_vs_cost_chart(df)
    c8 = create_age_vs_cost_scatter(df)
    c9 = create_bmi_vs_cost_scatter(df)
    c10 = create_risk_distribution_chart(df)
    c11 = create_risk_gauge_chart(55.0, 'MEDIUM')

    # Model diagnostics charts
    report = load_evaluation_report()
    best_name = report['best_model_name']
    model_data = report['models'][best_name]

    c12 = create_actual_vs_predicted_chart(report['y_true_sample'], model_data['y_pred_sample'], best_name)
    c13 = create_residual_plot(model_data['y_pred_sample'], model_data['residuals_sample'], best_name)
    c14 = create_error_distribution_chart(model_data['residuals_sample'], best_name)
    c15 = create_feature_importance_chart(model_data['feature_importances'], f"Feature Importance ({best_name})")

    # Scenario comparison chart
    scenario_df = pd.DataFrame([
        {'Scenario': 'Baseline', 'Predicted Annual Cost': 5000.0},
        {'Scenario': 'Smoker', 'Predicted Annual Cost': 25000.0}
    ])
    c16 = create_scenario_comparison_chart(scenario_df)

    charts = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
    for idx, c in enumerate(charts, 1):
        assert c is not None, f"Chart {idx} returned None"

    print(f"[OK] Verified all 16 Plotly visualization generators successfully.")

def test_data_explorer_operations():
    """Verify statistics computation, schema inspection, and CSV export."""
    proc_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')
    df = pd.read_csv(proc_path)

    # Stats
    stats = df.describe()
    assert not stats.empty

    # Nulls check
    assert df.isnull().sum().sum() == 0

    # CSV serialization
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    assert len(csv_bytes) > 1000

    print("[OK] Verified Data Explorer operations and CSV serialization.")

if __name__ == '__main__':
    test_inference_boundary_permutations()
    test_all_chart_generators()
    test_data_explorer_operations()
    print("\n[SUCCESS] Final stability and quality audit completed with 0 errors!")
