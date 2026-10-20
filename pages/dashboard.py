"""
Executive Dashboard Page
Displays high-level actuarial metrics, demographic distributions, interactive filters,
and dynamic Plotly visualizations.
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
    compute_kpis,
    create_cost_distribution_chart,
    create_cost_by_age_group_chart,
    create_cost_by_bmi_category_chart,
    create_cost_by_smoker_chart,
    create_cost_by_gender_chart,
    create_cost_by_region_chart,
    create_children_vs_cost_chart,
    create_age_vs_cost_scatter
)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load processed dataset with error handling."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'insurance_processed.csv')
    if not os.path.exists(data_path):
        raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'insurance_dataset.csv')
        if not os.path.exists(raw_path):
            st.error("⚠️ Dataset not found. Please train models or run data generation pipeline first.")
            return pd.DataFrame()
        from src.data_processing.preprocess import engineer_features
        return engineer_features(pd.read_csv(raw_path))
    return pd.read_csv(data_path)

def render_dashboard():
    # Hero Title Banner
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Executive Overview</span>
        <div class="hero-title">AI Insurance Cost & Risk Analytics</div>
        <div class="hero-subtitle">Predict • Analyze • Understand • Simulate</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.stop()

    # Filter Sidebar / Controls
    st.sidebar.markdown("### 🎛️ Dashboard Filters")
    st.sidebar.caption("Filter portfolio data across demographic dimensions.")

    # 1. Age Range
    min_age = int(df['age'].min())
    max_age = int(df['age'].max())
    selected_age = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # 2. Gender
    gender_options = sorted(list(df['sex'].unique()))
    selected_gender = st.sidebar.multiselect("Gender", options=gender_options, default=gender_options)

    # 3. BMI Range
    min_bmi = float(df['bmi'].min())
    max_bmi = float(df['bmi'].max())
    selected_bmi = st.sidebar.slider("BMI Range", min_value=round(min_bmi, 1), max_value=round(max_bmi, 1), value=(min_bmi, max_bmi), step=0.1)

    # 4. Smoking Status
    smoker_options = sorted(list(df['smoker'].unique()))
    selected_smoker = st.sidebar.multiselect("Smoking Status", options=smoker_options, default=smoker_options)

    # 5. Region
    region_options = sorted(list(df['region'].unique()))
    selected_region = st.sidebar.multiselect("Geographic Region", options=region_options, default=region_options)

    # 6. Children
    children_options = sorted(list(df['children'].unique()))
    selected_children = st.sidebar.multiselect("Number of Children / Dependents", options=children_options, default=children_options)

    # Apply Filters
    filtered_df = df[
        (df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1]) &
        (df['sex'].isin(selected_gender)) &
        (df['bmi'] >= selected_bmi[0]) & (df['bmi'] <= selected_bmi[1]) &
        (df['smoker'].isin(selected_smoker)) &
        (df['region'].isin(selected_region)) &
        (df['children'].isin(selected_children))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No customer records match the selected filter criteria. Please broaden your selection.")
        st.stop()

    # Calculate KPIs on filtered data
    kpis = compute_kpis(filtered_df)

    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
            <div class="kpi-sub">Active Policyholders</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Cost</div>
            <div class="kpi-value">${kpis['avg_cost']:,.0f}</div>
            <div class="kpi-sub">Annual Mean Premium</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Median Cost</div>
            <div class="kpi-value">${kpis['median_cost']:,.0f}</div>
            <div class="kpi-sub">50th Percentile Benchmark</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High-Risk Share</div>
            <div class="kpi-value" style="color:#ef4444;">{kpis['high_risk_pct']}%</div>
            <div class="kpi-sub">Score ≥ 70 Risk Tier</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI Cards Row 2
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Minimum Cost</div>
            <div class="kpi-value">${kpis['min_cost']:,.0f}</div>
            <div class="kpi-sub">Lowest Observed Premium</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Maximum Cost</div>
            <div class="kpi-value">${kpis['max_cost']:,.0f}</div>
            <div class="kpi-sub">Ceiling Claim Outlay</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Age</div>
            <div class="kpi-value">{kpis['avg_age']} yrs</div>
            <div class="kpi-sub">Cohort Chronological Mean</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average BMI</div>
            <div class="kpi-value">{kpis['avg_bmi']}</div>
            <div class="kpi-sub">Body Mass Index Metric</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Section
    tab1, tab2, tab3 = st.tabs(["📊 Cost Distributions & Demographics", "🔍 Lifestyle & Regional Analysis", "📈 Correlation & Trends"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_cost_distribution_chart(filtered_df), width='stretch')
        with c2:
            st.plotly_chart(create_cost_by_age_group_chart(filtered_df), width='stretch')

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(create_cost_by_bmi_category_chart(filtered_df), width='stretch')
        with c4:
            st.plotly_chart(create_cost_by_smoker_chart(filtered_df), width='stretch')

        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(create_cost_by_gender_chart(filtered_df), width='stretch')
        with c6:
            st.plotly_chart(create_cost_by_region_chart(filtered_df), width='stretch')

    with tab3:
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(create_children_vs_cost_chart(filtered_df), width='stretch')
        with c8:
            st.plotly_chart(create_age_vs_cost_scatter(filtered_df), width='stretch')

    # Download filtered dataset option
    st.markdown("---")
    st.download_button(
        label="📥 Download Filtered Portfolio Data (CSV)",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_insurance_portfolio.csv',
        mime='text/csv'
    )

    st.markdown("""
    <div class="disclaimer-card">
        <b>Data Integrity Notice:</b> This dashboard operates on a <b>Synthetic/Demo Dataset</b> generated for actuarial modeling and ML risk estimation demonstration. It is not derived from proprietary insurer internal policyholder files.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_dashboard()
