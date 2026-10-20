"""
Model Performance & ML Diagnostics Page
Compares Linear Regression, Random Forest, and Gradient Boosting algorithms,
displays regression performance benchmarks (MAE, MSE, RMSE, R²),
actual vs predicted scatter plots, residual diagnostics, and feature importances.
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
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def render_model_performance():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Machine Learning Operations (MLOps)</span>
        <div class="hero-title">Model Performance & Evaluation</div>
        <div class="hero-subtitle">Multi-Model Benchmarking, Residual Diagnostics & Statistical Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    report = load_evaluation_report()

    if not report:
        st.warning(
            "⚠️ Model evaluation report not found. "
            "Please ensure the training pipeline has executed and generated model artifacts."
        )
        st.stop()

    best_name = report.get('best_model_name', 'Gradient Boosting Regressor')
    criterion = report.get('selection_criterion', 'Highest Test R² and Lowest Test RMSE')
    models_data = report.get('models', {})
    y_true = report.get('y_true_sample', [])
    train_n = report.get('training_samples', '4,800')
    test_n = report.get('test_samples', '1,200')

    # Winner Banner
    st.markdown(f"""
    <div class="glass-panel" style="border-left: 6px solid #10b981; background: #f0fdf4;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 0.05em;">Winning Production Model</span>
        <h2 style="color: #065f46; margin: 4px 0 8px 0;">🏆 {best_name}</h2>
        <div style="font-size: 0.95rem; color: #166534; line-height: 1.5;">
            <b>Selection Rationale:</b> Selected automatically based on <b>{criterion}</b>.<br>
            <b>Sample Split:</b> {train_n} training records (80%) / {test_n} holdout test records (20%).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Model Comparison Table
    st.markdown("### 📊 Benchmark Comparison Table")
    st.caption("Empirical performance on holdout test partition (20% sample, random_state=42).")

    comparison_rows = []
    for m_name, m_info in models_data.items():
        metrics = m_info.get('metrics', {})
        r2_val = metrics.get('R2')
        rmse_val = metrics.get('RMSE')
        mae_val = metrics.get('MAE')
        mse_val = metrics.get('MSE')

        comparison_rows.append({
            'Model Architecture': m_name,
            'Test R² Score': f"{r2_val:.4f}" if r2_val is not None else "N/A",
            'RMSE ($)': f"${rmse_val:,.2f}" if rmse_val is not None else "N/A",
            'MAE ($)': f"${mae_val:,.2f}" if mae_val is not None else "N/A",
            'MSE ($²)': f"${mse_val:,.0f}" if mse_val is not None else "N/A",
            'Production Status': '🚀 ACTIVE (Best)' if m_name == best_name else 'Candidate'
        })

    df_comp = pd.DataFrame(comparison_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Interactive Diagnostic Visualizations for Chosen Model
    st.markdown("### 🔍 In-Depth Model Diagnostics & Residuals")

    selected_model_name = st.selectbox(
        "Select Model to Inspect Diagnostics:",
        options=list(models_data.keys()),
        index=list(models_data.keys()).index(best_name) if best_name in models_data else 0
    )

    curr_model_data = models_data.get(selected_model_name, {})
    y_pred = curr_model_data.get('y_pred_sample', [])
    residuals = curr_model_data.get('residuals_sample', [])
    feature_importances = curr_model_data.get('feature_importances', {})

    tab_diag1, tab_diag2, tab_diag3 = st.tabs([
        "🎯 Actual vs Predicted & Residuals",
        "🌲 Feature Importance & Explainability",
        "⚙️ Pipeline Management & Retraining"
    ])

    with tab_diag1:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.plotly_chart(
                create_actual_vs_predicted_chart(y_true, y_pred, selected_model_name),
                use_container_width=True
            )
        with d_col2:
            st.plotly_chart(
                create_residual_plot(y_pred, residuals, selected_model_name),
                use_container_width=True
            )

        st.plotly_chart(
            create_error_distribution_chart(residuals, selected_model_name),
            use_container_width=True
        )

    with tab_diag2:
        st.markdown("#### Global Feature Importance")
        st.caption("Quantifies the relative statistical contribution of each preprocessed feature toward minimizing model loss.")
        if feature_importances:
            st.plotly_chart(
                create_feature_importance_chart(feature_importances, f"Feature Importance ({selected_model_name})"),
                use_container_width=True
            )
        else:
            st.info("Feature importance data is unavailable for the selected model architecture.")

        st.markdown("""
        <div class="glass-panel">
            <h4>💡 Explainability Key Takeaways</h4>
            <ul>
                <li><b>Primary Predictor:</b> Smoker classification (one-hot indicator) accounts for the largest fraction of model variance reduction.</li>
                <li><b>Secondary Factor:</b> Continuous Body Mass Index (BMI), particularly with elevated variance above the 30.0 obesity threshold.</li>
                <li><b>Tertiary Factor:</b> Age chronologically scales expected claim baselines.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_diag3:
        st.markdown("#### Model Pipeline Operations")
        st.caption("Administrative retraining controls. Active production models should only be retrained intentionally.")

        with st.expander("⚠️ Retraining Controls", expanded=False):
            st.warning(
                "Triggering retraining will refit candidate algorithms (Linear Regression, Random Forest, Gradient Boosting) "
                "on the processed dataset and overwrite serialized artifacts in `models/`."
            )
            confirm_retrain = st.checkbox("I understand that this will overwrite existing model artifacts.")
            if confirm_retrain:
                if st.button("🔄 Execute End-to-End Retraining Pipeline", use_container_width=True):
                    with st.spinner("Executing model training pipeline..."):
                        train_and_evaluate_all()
                        st.success("Retraining complete! Model evaluation report refreshed.")
                        st.rerun()

    # Evaluation Limitations Section
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-card">
        <b>Model Evaluation Limitations:</b>
        <ul>
            <li>Metrics reflect out-of-sample evaluation on an 80/20 train-test split of the synthetic dataset.</li>
            <li>Real-world policyholder distributions may experience covariate shift (e.g., regional healthcare inflation, changing policy benefits).</li>
            <li>Feature importances indicate predictive utility within this model architecture and do not establish direct biological causality.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_model_performance()
