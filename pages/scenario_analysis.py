"""
What-If Scenario Analysis Page
Enables comparative actuarial simulation across multiple customer cohorts,
real-time sensitivity adjustment, baseline delta calculations,
and multi-scenario comparison charts.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction_helper import get_predictor
from src.analytics.analytics_helper import create_scenario_comparison_chart

def render_scenario_analysis():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Actuarial Simulation</span>
        <div class="hero-title">What-If Scenario Analysis</div>
        <div class="hero-subtitle">Simulate Cohorts, Contrast Risk Profiles & Evaluate Baseline Deviations</div>
    </div>
    """, unsafe_allow_html=True)

    predictor = get_predictor()
    if not predictor.is_loaded:
        st.error("⚠️ Trained model artifact not found. Please train models first.")
        st.stop()

    tab_preset, tab_live = st.tabs([
        "📋 Multi-Scenario Cohort Comparison",
        "🎛️ Real-Time Baseline Sensitivity Simulator"
    ])

    # TAB 1: Preset & Custom Multi-Scenario Cohorts
    with tab_preset:
        st.markdown("### 🏢 Cohort Risk & Premium Matrix")
        st.caption("Evaluate and benchmark distinct risk archetypes against the baseline profile.")

        # Default predefined scenarios
        default_scenarios = [
            {
                'name': 'Scenario A: Young & Healthy',
                'age': 25, 'sex': 'female', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'southwest'
            },
            {
                'name': 'Scenario B: Mid-Age Overweight Smoker',
                'age': 45, 'sex': 'male', 'bmi': 30.0, 'children': 2, 'smoker': 'yes', 'region': 'southeast'
            },
            {
                'name': 'Scenario C: Senior Obese Smoker',
                'age': 60, 'sex': 'female', 'bmi': 35.0, 'children': 1, 'smoker': 'yes', 'region': 'northeast'
            }
        ]

        # Allow user to add or modify custom scenario
        with st.expander("➕ Configure Custom Scenario D", expanded=False):
            sc_c1, sc_c2, sc_c3 = st.columns(3)
            with sc_c1:
                custom_name = st.text_input("Scenario Name", value="Scenario D: Custom Profile")
                custom_age = st.slider("Custom Age", 18, 80, 52)
                custom_gender = st.selectbox("Custom Gender", ["Male", "Female"], index=0)
            with sc_c2:
                custom_bmi = st.slider("Custom BMI", 15.0, 50.0, 28.4, step=0.1)
                custom_children = st.selectbox("Custom Children", [0, 1, 2, 3, 4, 5], index=1)
            with sc_c3:
                custom_smoker = st.selectbox("Custom Smoker", ["No", "Yes"], index=0)
                custom_region = st.selectbox("Custom Region", ["Southwest", "Southeast", "Northwest", "Northeast"], index=1)

            include_custom = st.checkbox("Include Scenario D in Comparison", value=True)

        all_scenarios = list(default_scenarios)
        if include_custom:
            all_scenarios.append({
                'name': custom_name,
                'age': custom_age,
                'sex': custom_gender.lower(),
                'bmi': custom_bmi,
                'children': custom_children,
                'smoker': custom_smoker.lower(),
                'region': custom_region.lower()
            })

        # Run predictions for every scenario
        scenario_results = []
        for s in all_scenarios:
            pred = predictor.predict_single(
                age=s['age'],
                sex=str(s['sex']),
                bmi=s['bmi'],
                children=s['children'],
                smoker=str(s['smoker']),
                region=str(s['region'])
            )
            scenario_results.append({
                'Scenario': s['name'],
                'Age': s['age'],
                'Gender': str(s['sex']).capitalize(),
                'BMI': s['bmi'],
                'Children': s['children'],
                'Smoker': str(s['smoker']).capitalize(),
                'Region': str(s['region']).capitalize(),
                'Predicted Annual Cost': pred['predicted_annual_cost'],
                'Monthly Cost': pred['predicted_monthly_cost'],
                'Risk Score': pred['risk_score'],
                'Risk Category': pred['risk_category']
            })

        df_scenarios = pd.DataFrame(scenario_results)

        # Baseline is Scenario A (first scenario)
        baseline_cost = df_scenarios.loc[0, 'Predicted Annual Cost']

        df_scenarios['Difference from baseline'] = df_scenarios['Predicted Annual Cost'] - baseline_cost
        df_scenarios['Percentage Difference'] = (
            (df_scenarios['Difference from baseline'] / baseline_cost) * 100.0
        ).round(1)

        # Formatted Display Table
        df_display = df_scenarios.copy()
        df_display['Annual Cost'] = df_display['Predicted Annual Cost'].apply(lambda x: f"${x:,.2f}")
        df_display['Monthly Cost'] = df_display['Monthly Cost'].apply(lambda x: f"${x:,.2f}")
        df_display['Difference ($)'] = df_display['Difference from baseline'].apply(lambda x: f"{'+' if x >= 0 else ''}${x:,.2f}")
        df_display['Difference (%)'] = df_display['Percentage Difference'].apply(lambda x: f"{'+' if x >= 0 else ''}{x}%")

        table_cols = ['Scenario', 'Age', 'Gender', 'BMI', 'Smoker', 'Children', 'Region', 'Annual Cost', 'Monthly Cost', 'Difference ($)', 'Difference (%)', 'Risk Category']
        st.dataframe(df_display[table_cols], use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Comparison Chart
        st.plotly_chart(create_scenario_comparison_chart(df_scenarios), use_container_width=True)

        # CSV Download
        csv_data = df_scenarios.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scenario Comparison (CSV)",
            data=csv_data,
            file_name="insurance_what_if_scenarios.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TAB 2: Real-Time Sensitivity Adjustment
    with tab_live:
        st.markdown("### 🎛️ Real-Time Baseline Sensitivity Tool")
        st.caption("Instantly modify individual applicant traits to isolate and measure exact marginal cost impacts.")

        col_base, col_sim = st.columns(2)

        with col_base:
            st.markdown("#### 1️⃣ Baseline Applicant")
            b_age = st.slider("Baseline Age", 18, 70, 30, key="b_age")
            b_bmi = st.slider("Baseline BMI", 16.0, 45.0, 23.5, step=0.1, key="b_bmi")
            b_smoker = st.selectbox("Baseline Smoking Status", ["No", "Yes"], index=0, key="b_smoker")
            b_children = st.selectbox("Baseline Children", [0, 1, 2, 3, 4, 5], index=0, key="b_kids")
            b_gender = st.selectbox("Baseline Gender", ["Male", "Female"], key="b_gender")
            b_region = st.selectbox("Baseline Region", ["Southwest", "Southeast", "Northwest", "Northeast"], key="b_region")

        with col_sim:
            st.markdown("#### 2️⃣ Simulated Variant")
            s_age = st.slider("Simulated Age", 18, 70, 48, key="s_age")
            s_bmi = st.slider("Simulated BMI", 16.0, 45.0, 31.2, step=0.1, key="s_bmi")
            s_smoker = st.selectbox("Simulated Smoking Status", ["No", "Yes"], index=1, key="s_smoker")
            s_children = st.selectbox("Simulated Children", [0, 1, 2, 3, 4, 5], index=2, key="s_kids")
            s_gender = st.selectbox("Simulated Gender", ["Male", "Female"], key="s_gender")
            s_region = st.selectbox("Simulated Region", ["Southwest", "Southeast", "Northwest", "Northeast"], key="s_region")

        pred_base = predictor.predict_single(b_age, b_gender.lower(), b_bmi, b_children, b_smoker.lower(), b_region.lower())
        pred_sim = predictor.predict_single(s_age, s_gender.lower(), s_bmi, s_children, s_smoker.lower(), s_region.lower())

        cost_diff = pred_sim['predicted_annual_cost'] - pred_base['predicted_annual_cost']
        pct_diff = ((cost_diff / pred_base['predicted_annual_cost']) * 100.0) if pred_base['predicted_annual_cost'] > 0 else 0.0

        st.markdown("---")
        st.markdown("#### ⚡ Real-Time Marginal Impact Analysis")

        delta_c1, delta_c2, delta_c3, delta_c4 = st.columns(4)

        with delta_c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Baseline Annual Cost</div>
                <div class="kpi-value">${pred_base['predicted_annual_cost']:,.2f}</div>
                <div class="kpi-sub">${pred_base['predicted_monthly_cost']:,.2f}/mo</div>
            </div>
            """, unsafe_allow_html=True)

        with delta_c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Simulated Annual Cost</div>
                <div class="kpi-value" style="color: #0284c7;">${pred_sim['predicted_annual_cost']:,.2f}</div>
                <div class="kpi-sub">${pred_sim['predicted_monthly_cost']:,.2f}/mo</div>
            </div>
            """, unsafe_allow_html=True)

        diff_color = "#ef4444" if cost_diff > 0 else "#10b981"
        with delta_c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cost Delta ($)</div>
                <div class="kpi-value" style="color: {diff_color};">{'+' if cost_diff >= 0 else ''}${cost_diff:,.2f}</div>
                <div class="kpi-sub">Annual variance</div>
            </div>
            """, unsafe_allow_html=True)

        with delta_c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Relative Deviation (%)</div>
                <div class="kpi-value" style="color: {diff_color};">{'+' if pct_diff >= 0 else ''}{pct_diff:.1f}%</div>
                <div class="kpi-sub">Versus baseline profile</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_scenario_analysis()
