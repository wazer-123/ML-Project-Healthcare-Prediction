"""
Data Explorer Page
Provides comprehensive dataset inspection, schema auditing, missing-value summaries,
dynamic filtering, searchable record exploration, and CSV export.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analytics.analytics_helper import apply_chart_style


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed insurance dataset or fallback to raw dataset."""
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


def render_data_explorer():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Data Quality & Governance</span>
        <div class="hero-title">Data Explorer</div>
        <div class="hero-subtitle">Dataset Auditing, Schema Inspection & Custom Query Visualizations</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.error("⚠️ Dataset not available. Please verify data pipeline generation.")
        st.stop()

    st.markdown("""
    <div class="disclaimer-card">
        <b>Data Provenance:</b> Synthetic actuarial dataset generated for machine learning benchmarking.
        All demographic records are synthesized and do not correspond to individual private policyholders.
    </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Overview Metrics
    st.markdown("### 📋 Portfolio Metadata")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">Total rows in dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Features</div>
            <div class="kpi-value">{df.shape[1]}</div>
            <div class="kpi-sub">Columns in schema</div>
        </div>
        """, unsafe_allow_html=True)

    missing_cnt = int(df.isnull().sum().sum())
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Missing Values</div>
            <div class="kpi-value" style="color: {'#10b981' if missing_cnt == 0 else '#ef4444'};">{missing_cnt}</div>
            <div class="kpi-sub">Complete data hygiene</div>
        </div>
        """, unsafe_allow_html=True)

    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 1)
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Memory Footprint</div>
            <div class="kpi-value">{mem_kb:,.1f} KB</div>
            <div class="kpi-sub">In-memory footprint</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Data Explorer
    tab_view, tab_filter, tab_schema, tab_stats, tab_custom = st.tabs([
        "📄 Dataset Preview",
        "🔍 Filter & Search Records",
        "🗂️ Schema & Data Types",
        "📈 Descriptive Statistics",
        "🎨 Custom Visual Query Builder"
    ])

    with tab_view:
        st.markdown("#### Tabular Records Preview")
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            sample_rows = st.slider("Display row limit", 10, min(1000, len(df)), 50, step=10)
        with col_ctrl2:
            st.write("")
            st.write(f"Showing first **{sample_rows}** of **{len(df):,}** records")

        st.dataframe(df.head(sample_rows), use_container_width=True)

        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="insurance_complete_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_filter:
        st.markdown("#### Dynamic Record Filtering & Query")
        st.caption("Apply interactive filters to inspect specific cohort subsets without altering the source data.")

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            filter_sex = st.multiselect(
                "Filter Gender",
                options=sorted(df['sex'].unique()),
                default=sorted(df['sex'].unique())
            )
        with fcol2:
            filter_smoker = st.multiselect(
                "Filter Smoker",
                options=sorted(df['smoker'].unique()),
                default=sorted(df['smoker'].unique())
            )
        with fcol3:
            filter_region = st.multiselect(
                "Filter Region",
                options=sorted(df['region'].unique()),
                default=sorted(df['region'].unique())
            )

        fcol4, fcol5 = st.columns(2)
        with fcol4:
            age_range = st.slider(
                "Filter Age Range",
                min_value=int(df['age'].min()),
                max_value=int(df['age'].max()),
                value=(int(df['age'].min()), int(df['age'].max()))
            )
        with fcol5:
            bmi_range = st.slider(
                "Filter BMI Range",
                min_value=float(df['bmi'].min()),
                max_value=float(df['bmi'].max()),
                value=(float(df['bmi'].min()), float(df['bmi'].max())),
                step=0.1
            )

        # Apply non-destructive filtering
        filtered_view = df[
            (df['sex'].isin(filter_sex)) &
            (df['smoker'].isin(filter_smoker)) &
            (df['region'].isin(filter_region)) &
            (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
            (df['bmi'] >= bmi_range[0]) & (df['bmi'] <= bmi_range[1])
        ]

        st.info(f"Filtered result contains **{len(filtered_view):,}** matching records ({len(filtered_view)/len(df)*100:.1f}% of total).")
        st.dataframe(filtered_view.head(200), use_container_width=True)

        if not filtered_view.empty:
            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=filtered_view.to_csv(index=False).encode('utf-8'),
                file_name="insurance_filtered_query.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab_schema:
        st.markdown("#### Schema & Data Type Inspection")
        schema_info = []
        for col in df.columns:
            schema_info.append({
                'Column Name': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': int(df[col].count()),
                'Null Count': int(df[col].isnull().sum()),
                'Unique Values': int(df[col].nunique()),
                'Sample Value': str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ''
            })
        st.dataframe(pd.DataFrame(schema_info), use_container_width=True, hide_index=True)

    with tab_stats:
        st.markdown("#### Numerical Feature Statistics")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            st.markdown("#### Categorical Dimension Distributions")
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)

    with tab_custom:
        st.markdown("#### 🎨 Custom Visual Query Builder")
        st.caption("Select arbitrary dimensions from the dataset to compose on-the-fly analytical charts.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            chart_type = st.selectbox("Chart Type", ["Scatter", "Box", "Histogram", "Bar", "Violin"])
        with c2:
            x_col = st.selectbox("X-Axis Feature", options=list(df.columns), index=0)
        with c3:
            numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
            y_col = st.selectbox(
                "Y-Axis Feature",
                options=numeric_cols,
                index=min(len(numeric_cols)-1, 3)
            )
        with c4:
            color_col = st.selectbox(
                "Color Grouping",
                options=["None"] + list(df.columns),
                index=5 if len(df.columns) > 5 else 0
            )

        color_arg = None if color_col == "None" else color_col

        try:
            fig = None
            sample_df = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df

            if chart_type == "Scatter":
                fig = px.scatter(sample_df, x=x_col, y=y_col, color=color_arg, opacity=0.7)
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y_col, color=color_arg)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_arg, marginal='box')
            elif chart_type == "Bar":
                if color_arg and color_arg != x_col:
                    agg_df = df.groupby([x_col, color_arg])[y_col].mean().reset_index()
                    fig = px.bar(agg_df, x=x_col, y=y_col, color=color_arg, barmode='group')
                else:
                    agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(agg_df, x=x_col, y=y_col)
            elif chart_type == "Violin":
                fig = px.violin(df, x=x_col, y=y_col, color=color_arg, box=True)

            if fig:
                st.plotly_chart(
                    apply_chart_style(fig, f"{chart_type}: {x_col} vs {y_col}"),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error constructing custom visualization: {e}")


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_data_explorer()
