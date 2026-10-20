"""
Cost & Demographic Analytics Page
Provides exploratory analyses covering age, BMI, smoking status,
gender, region, dependents, correlation structure, and statistical summaries.
Strictly maintains non-causal statistical language.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analytics.analytics_helper import (
    create_cost_distribution_chart,
    create_age_vs_cost_scatter,
    create_bmi_vs_cost_scatter,
    create_children_vs_cost_chart,
    create_cost_by_gender_chart,
    create_cost_by_smoker_chart,
    create_cost_by_region_chart,
    create_cost_by_age_group_chart,
    create_cost_by_bmi_category_chart,
    create_risk_distribution_chart,
    create_correlation_heatmap
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed insurance dataset or process the raw dataset."""
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "insurance_processed.csv")
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "insurance_dataset.csv")

    try:
        if os.path.isfile(data_path):
            return pd.read_csv(data_path)

        if os.path.isfile(raw_path):
            from src.data_processing.preprocess import engineer_features
            return engineer_features(pd.read_csv(raw_path))

        st.error("Dataset not found. Checked both processed and raw dataset paths.")
        return pd.DataFrame()

    except Exception as exc:
        st.error(f"Dataset loading failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def render_analytics():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Demographic & Cost Intelligence</span>
        <div class="hero-title">Cost & Demographic Analytics</div>
        <div class="hero-subtitle">Multi-Dimensional Exploratory Analysis & Statistical Associations</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.error("⚠️ Dataset not available. Please ensure the data generation pipeline has executed.")
        st.stop()

    st.markdown("### 📊 Actuarial Exploration Matrices")
    st.caption(
        "Investigate empirical cost distributions across demographic variables and lifestyle determinants. "
        "All visual relationships reflect statistical associations in the training sample, not causal proofs."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Continuous Trends & Scatter",
        "👥 Demographic & Regional Breakdown",
        "🛡️ Categorical & Risk Segmentation",
        "📐 Correlation & Data Summary"
    ])

    with tab1:
        st.markdown("#### Continuous Feature Associations: Age, BMI & Dependents")
        st.caption("Visualizing distribution patterns and bivariate scatter plots.")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_cost_distribution_chart(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_age_vs_cost_scatter(df), use_container_width=True)

        col1b, col2b = st.columns(2)
        with col1b:
            st.plotly_chart(create_bmi_vs_cost_scatter(df), use_container_width=True)
        with col2b:
            st.plotly_chart(create_children_vs_cost_chart(df), use_container_width=True)

    with tab2:
        st.markdown("#### Categorical Profiles: Smoking, Gender & Geography")
        st.caption("Evaluating demographic cohorts across categorical partitions.")

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_cost_by_smoker_chart(df), use_container_width=True)
        with col4:
            st.plotly_chart(create_cost_by_gender_chart(df), use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(create_cost_by_region_chart(df), use_container_width=True)
        with col6:
            st.plotly_chart(create_cost_by_age_group_chart(df), use_container_width=True)

    with tab3:
        st.markdown("#### Risk Stratification & Clinical Categorization")
        st.caption("Grouping records by World Health Organization BMI categories and heuristic risk tiers.")

        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(create_cost_by_bmi_category_chart(df), use_container_width=True)
        with col8:
            st.plotly_chart(create_risk_distribution_chart(df), use_container_width=True)

        st.markdown("##### 💡 Empirical Actuarial Observations")
        st.markdown(
            "- **Smoking Association:** In this dataset, smoking status is associated with an elevated cost tier "
            "(median charges exceed $30,000 for smokers vs ~$9,000 for non-smokers).\n"
            "- **BMI Threshold:** Observations with BMI ≥ 30 (obesity threshold) display higher variance, "
            "particularly when combined with active smoking.\n"
            "- **Age Gradient:** Charges display a steady positive association with chronological age across all cohorts.\n"
            "- **Statistical Distinction:** These findings represent observable empirical correlations. "
            "They illustrate dataset associations rather than deterministic causal mechanisms for individual patients."
        )

    with tab4:
        st.markdown("#### Statistical Summaries & Feature Correlation")
        st.caption("Pearson correlation coefficients and parametric summary statistics.")

        corr_col, stat_col = st.columns([1.1, 1.2])

        with corr_col:
            st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

        with stat_col:
            st.markdown("##### 📋 Descriptive Statistics (Numerical Features)")
            num_cols = ['age', 'bmi', 'children', 'charges']
            if 'risk_score' in df.columns:
                num_cols.append('risk_score')

            desc_df = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
            desc_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']
            st.dataframe(desc_df.round(2), use_container_width=True)

            st.markdown("##### 🔍 Missing-Value Audit")
            missing_series = df[num_cols].isna().sum()
            missing_df = pd.DataFrame({
                'Feature': missing_series.index,
                'Missing Values': missing_series.values,
                'Completeness': [f"{(1 - v/len(df))*100:.1f}%" for v in missing_series.values]
            })
            st.dataframe(missing_df, use_container_width=True, hide_index=True)


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_analytics()
