"""
Cost & Demographic Analytics Page
Provides deep multi-dimensional exploratory charts covering age, BMI,
smoking status, gender, region, dependents, and risk distributions.
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
    create_age_vs_cost_scatter,
    create_bmi_vs_cost_scatter,
    create_children_vs_cost_chart,
    create_cost_by_gender_chart,
    create_cost_by_smoker_chart,
    create_cost_by_region_chart,
    create_cost_by_age_group_chart,
    create_cost_by_bmi_category_chart,
    create_risk_distribution_chart
)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed insurance dataset or process the raw dataset."""

    data_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "insurance_processed.csv"
    )

    raw_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "insurance_dataset.csv"
    )

    try:
        # Prefer the processed dataset
        if os.path.isfile(data_path):
            return pd.read_csv(data_path)

        # Fall back to the raw dataset
        if os.path.isfile(raw_path):
            from src.data_processing.preprocess import engineer_features

            return engineer_features(pd.read_csv(raw_path))

        st.error(
            "Dataset not found. Checked both processed and raw dataset paths."
        )
        return pd.DataFrame()

    except Exception as exc:
        st.error(
            f"Dataset loading failed: {type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()

def render_analytics():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Demographic & Cost Intelligence</span>
        <div class="hero-title">Cost & Demographic Analytics</div>
        <div class="hero-subtitle">Multi-Dimensional Exploratory Analysis Across Actuarial Factors</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.error("⚠️ Dataset not available. Please ensure the data generation pipeline has executed.")
        st.stop()

    st.markdown("### 📊 Actuarial Exploration Matrices")
    st.caption("Investigate empirical cost distributions across demographic variables and lifestyle determinants.")

    tab1, tab2, tab3 = st.tabs([
        "🔬 Continuous Scatter & Trends",
        "👥 Demographic & Geographic Breakdown",
        "🛡️ Categorical & Risk Segmentation"
    ])

    with tab1:
        st.markdown("#### Continuous Drivers: Age, BMI & Dependents")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_age_vs_cost_scatter(df), width='stretch')
        with col2:
            st.plotly_chart(create_bmi_vs_cost_scatter(df), width='stretch')

        st.plotly_chart(create_children_vs_cost_chart(df), width='stretch')

    with tab2:
        st.markdown("#### Categorical Drivers: Lifestyle, Gender & Regional Dynamics")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_cost_by_smoker_chart(df), width='stretch')
        with col4:
            st.plotly_chart(create_cost_by_gender_chart(df), width='stretch')

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(create_cost_by_region_chart(df), width='stretch')
        with col6:
            st.plotly_chart(create_cost_by_age_group_chart(df), width='stretch')

    with tab3:
        st.markdown("#### Risk & Clinical Grouping Comparisons")
        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(create_cost_by_bmi_category_chart(df), width='stretch')
        with col8:
            st.plotly_chart(create_risk_distribution_chart(df), width='stretch')

        st.markdown("##### 💡 Key Actuarial Observations")
        st.markdown(
            "- **Smoking Multiplier:** Active smoking creates a bifurcated distribution, "
            "elevating median charges by more than **300%** across comparable age brackets.\n"
            "- **Obesity Interaction:** The combination of BMI ≥ 30 and smoking demonstrates "
            "a non-linear compounding cost effect.\n"
            "- **Linear Age Gradient:** Within non-smoker populations, costs exhibit a "
            "consistent, predictable upward slope with chronological age."
        )

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_analytics()
