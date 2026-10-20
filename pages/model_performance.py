"""
Model Performance & ML Diagnostics Page
Compares Linear Regression, Random Forest, and Gradient Boosting algorithms,
displays regression performance benchmarks (MAE, MSE, RMSE, R²),
actual vs predicted scatter plots, residual distributions, and feature importances.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analytics.analytics_helper import (
    create_actual_vs_predicted_chart,
    create_residual_plot,
    create_error_distribution_chart,
    create_feature_importance_chart
)
from src.models.train_models import train_and_evaluate_all

def load_evaluation_report() -> dict:
    """Load cached model evaluation report from json."""
    report_path = os.path.join(PROJECT_ROOT, 'models', 'model_evaluation_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return json.load(f)
    return {}

def render_model_performance():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Machine Learning Operations (MLOps)</span>
        <div class="hero-title">Model Performance & Evaluation</div>
        <div class="hero-subtitle">Multi-Model Benchmarking, Residual Diagnostics & Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    report = load_evaluation_report()

    # Retrain button in header
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr2:
        if st.button("🔄 Retrain All Models", width='stretch'):
            with st.spinner("Executing end-to-end multi-model training pipeline..."):
                train_and_evaluate_all()
                st.success("Retraining complete!")
                st.rerun()

    if not report:
        st.warning("⚠️ Model evaluation report not found. Please click '🔄 Retrain All Models' above to execute the training pipeline and generate performance metrics.")
        st.stop()

    best_name = report.get('best_model_name', 'Gradient Boosting Regressor')
    criterion = report.get('selection_criterion', 'Highest Test R² and Lowest Test RMSE')
    models_data = report.get('models', {})
    y_true = report.get('y_true_sample', [])

    # Winner Banner
    st.markdown(f"""
    <div class="glass-panel" style="border-left: 6px solid #10b981; background: #f0fdf4;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 0.05em;">Winning Production Model</span>
        <h2 style="color: #065f46; margin: 4px 0 8px 0;">🏆 {best_name}</h2>
        <div style="font-size: 0.95rem; color: #166534;">
            <b>Selection Rationale:</b> Selected automatically based on <b>{criterion}</b>. Deployed to production pipeline via Joblib serialization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Model Comparison Table
    st.markdown("### 📊 Benchmark Comparison Table")
    comparison_rows = []
    for m_name, m_info in models_data.items():
        metrics = m_info.get('metrics', {})
        comparison_rows.append({
            'Model Architecture': m_name,
            'Test R² Score': metrics.get('R2'),
            'RMSE ($)': f"${metrics.get('RMSE'):,.2f}",
            'MAE ($)': f"${metrics.get('MAE'):,.2f}",
            'MSE ($²)': f"${metrics.get('MSE'):,.0f}",
            'Production Status': '🚀 ACTIVE (Best)' if m_name == best_name else 'Candidate'
        })

    df_comp = pd.DataFrame(comparison_rows).sort_values(by='Test R² Score', ascending=False)
    st.dataframe(df_comp, width='stretch', hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Interactive Diagnostic Visualizations for Chosen Model
    st.markdown("### 🔍 In-Depth Model Diagnostics & Residuals")

    selected_model_name = st.selectbox(
        "Select Model to Inspect Diagnostics:",
        options=list(models_data.keys()),
        index=list(models_data.keys()).index(best_name) if best_name in models_data else 0
    )

    curr_model_data = models_data[selected_model_name]
    y_pred = curr_model_data.get('y_pred_sample', [])
    residuals = curr_model_data.get('residuals_sample', [])
    feature_importances = curr_model_data.get('feature_importances', {})

    # Diagnostic charts
    tab_diag1, tab_diag2 = st.tabs(["🎯 Actual vs Predicted & Residuals", "🌲 Feature Importance & Explainability"])

    with tab_diag1:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.plotly_chart(create_actual_vs_predicted_chart(y_true, y_pred, selected_model_name), width='stretch')
        with d_col2:
            st.plotly_chart(create_residual_plot(y_pred, residuals, selected_model_name), width='stretch')

        st.plotly_chart(create_error_distribution_chart(residuals, selected_model_name), width='stretch')

    with tab_diag2:
        st.markdown("#### Global Feature Importance")
        st.caption("Quantifies the relative statistical contribution of each preprocessed feature toward minimizing variance.")
        st.plotly_chart(create_feature_importance_chart(feature_importances, f"Feature Importance ({selected_model_name})"), width='stretch')

        st.markdown("""
        <div class="glass-panel">
            <h4>💡 Explainability Key Takeaways</h4>
            <ul>
                <li><b>Primary Driver:</b> Smoker classification (and associated indicator interactions) accounts for the largest fraction of split-criterion gain.</li>
                <li><b>Secondary Driver:</b> Continuous Body Mass Index (BMI), especially above the 30.0 obesity threshold.</li>
                <li><b>Tertiary Driver:</b> Age chronologically scales claim baseline expectations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_model_performance()
