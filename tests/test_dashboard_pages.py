"""
Phase 5 Comprehensive Dashboard Page-by-Page Automated Validation Suite
Tests every individual page logic, calculation, filter, chart generation,
and data table operation across all 7 dashboard sections.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pages.dashboard import load_data as load_dashboard_data
from pages.analytics import load_data as load_analytics_data
from pages.data_explorer import load_data as load_explorer_data
from pages.model_performance import load_evaluation_report
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

def test_page_1_dashboard():
    print("Testing PAGE 1: Executive Dashboard...")
    df = load_dashboard_data()
    assert not df.empty, "Dashboard data must not be empty"
    
    # 1. Filter simulation
    filtered = df[(df['age'] >= 25) & (df['age'] <= 55) & (df['sex'] == 'female') & (df['smoker'] == 'no')]
    assert not filtered.empty, "Filtered dataframe should contain matching rows"
    
    # 2. KPI calculations
    kpis = compute_kpis(filtered)
    assert kpis['total_customers'] == len(filtered)
    assert kpis['avg_cost'] > 0
    assert kpis['min_cost'] <= kpis['median_cost'] <= kpis['max_cost']
    assert 0 <= kpis['high_risk_pct'] <= 100
    
    # 3. Charts
    c1 = create_cost_distribution_chart(filtered)
    c2 = create_cost_by_age_group_chart(filtered)
    c3 = create_cost_by_bmi_category_chart(filtered)
    c4 = create_cost_by_smoker_chart(filtered)
    c5 = create_cost_by_gender_chart(filtered)
    c6 = create_cost_by_region_chart(filtered)
    c7 = create_children_vs_cost_chart(filtered)
    c8 = create_age_vs_cost_scatter(filtered)
    for c in [c1, c2, c3, c4, c5, c6, c7, c8]:
        assert c is not None, "Dashboard chart cannot be None"
    print("  -> Page 1 Executive Dashboard: PASS")

def test_page_2_analytics():
    print("Testing PAGE 2: Cost & Demographic Analytics...")
    df = load_analytics_data()
    assert not df.empty
    
    # Continuous scatter
    s1 = create_age_vs_cost_scatter(df)
    s2 = create_bmi_vs_cost_scatter(df)
    s3 = create_children_vs_cost_chart(df)
    
    # Categorical breakdown
    s4 = create_cost_by_smoker_chart(df)
    s5 = create_cost_by_gender_chart(df)
    s6 = create_cost_by_region_chart(df)
    s7 = create_cost_by_age_group_chart(df)
    
    # Risk segmentation
    s8 = create_cost_by_bmi_category_chart(df)
    s9 = create_risk_distribution_chart(df)
    
    for s in [s1, s2, s3, s4, s5, s6, s7, s8, s9]:
        assert s is not None, "Analytics chart cannot be None"
    print("  -> Page 2 Analytics: PASS")

def test_page_3_predictor():
    print("Testing PAGE 3: Insurance Cost Predictor...")
    p = get_predictor()
    res = p.predict_single(age=35, sex='female', bmi=26.4, children=1, smoker='no', region='northwest')
    
    assert res['predicted_annual_cost'] > 0
    assert res['predicted_monthly_cost'] > 0
    assert 0 <= res['risk_score'] <= 100
    assert res['risk_category'] in ['LOW', 'MEDIUM', 'HIGH']
    assert len(res['explanation_insights']) >= 3
    
    gauge = create_risk_gauge_chart(res['risk_score'], res['risk_category'])
    assert gauge is not None
    print("  -> Page 3 Predictor: PASS")

def test_page_4_risk_profile():
    print("Testing PAGE 4: Customer Risk Profile...")
    p = get_predictor()
    # Baseline demo profile
    prof = p.predict_single(45, 'male', 29.8, 2, 'no', 'southwest')
    assert prof['age_category'] == '40-49'
    assert prof['bmi_category'] == 'Overweight'
    
    gauge = create_risk_gauge_chart(prof['risk_score'], prof['risk_category'])
    assert gauge is not None
    print("  -> Page 4 Risk Profile: PASS")

def test_page_5_scenario_analysis():
    print("Testing PAGE 5: What-If Scenario Analysis...")
    p = get_predictor()
    scenarios = [
        {'name': 'Young Healthy', 'age': 25, 'sex': 'female', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'southwest'},
        {'name': 'Mid-Age Overweight Smoker', 'age': 45, 'sex': 'male', 'bmi': 30.0, 'children': 2, 'smoker': 'yes', 'region': 'southeast'},
        {'name': 'Senior Obese Smoker', 'age': 60, 'sex': 'female', 'bmi': 35.0, 'children': 1, 'smoker': 'yes', 'region': 'northeast'}
    ]
    
    results = []
    for s in scenarios:
        pred = p.predict_single(s['age'], s['sex'], s['bmi'], s['children'], s['smoker'], s['region'])
        results.append({
            'Scenario': s['name'],
            'Predicted Annual Cost': pred['predicted_annual_cost'],
            'Monthly Cost': pred['predicted_monthly_cost']
        })
    df_sc = pd.DataFrame(results)
    baseline_cost = df_sc.loc[0, 'Predicted Annual Cost']
    df_sc['Difference from baseline'] = df_sc['Predicted Annual Cost'] - baseline_cost
    df_sc['Percentage Difference'] = ((df_sc['Difference from baseline'] / baseline_cost) * 100.0).round(1)
    
    assert len(df_sc) == 3
    assert df_sc.loc[0, 'Difference from baseline'] == 0.0
    chart = create_scenario_comparison_chart(df_sc)
    assert chart is not None
    print("  -> Page 5 Scenario Analysis: PASS")

def test_page_6_model_performance():
    print("Testing PAGE 6: Model Performance & Diagnostics...")
    rep = load_evaluation_report()
    assert 'best_model_name' in rep
    assert 'models' in rep
    
    best_name = rep['best_model_name']
    mdata = rep['models'][best_name]
    y_true = rep['y_true_sample']
    y_pred = mdata['y_pred_sample']
    resids = mdata['residuals_sample']
    importances = mdata['feature_importances']
    
    c1 = create_actual_vs_predicted_chart(y_true, y_pred, best_name)
    c2 = create_residual_plot(y_pred, resids, best_name)
    c3 = create_error_distribution_chart(resids, best_name)
    c4 = create_feature_importance_chart(importances, f"Importance ({best_name})")
    
    for c in [c1, c2, c3, c4]:
        assert c is not None
    print("  -> Page 6 Model Performance: PASS")

def test_page_7_data_explorer():
    print("Testing PAGE 7: Data Explorer...")
    df = load_explorer_data()
    assert not df.empty
    
    # Metadata metrics
    assert len(df) > 0
    assert df.shape[1] >= 7
    assert df.isnull().sum().sum() == 0
    
    # Schema inspection
    schema_info = [{'col': c, 'dtype': str(df[c].dtype)} for c in df.columns]
    assert len(schema_info) == df.shape[1]
    
    # Statistical summary
    desc = df.describe()
    assert not desc.empty
    
    # CSV download serialization
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    assert len(csv_bytes) > 0
    print("  -> Page 7 Data Explorer: PASS")

if __name__ == '__main__':
    test_page_1_dashboard()
    test_page_2_analytics()
    test_page_3_predictor()
    test_page_4_risk_profile()
    test_page_5_scenario_analysis()
    test_page_6_model_performance()
    test_page_7_data_explorer()
    print("\n[ALL 7 PAGES VALIDATED SUCCESSFULLY WITH 0 ERRORS]")
