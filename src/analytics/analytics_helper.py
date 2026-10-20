"""
Analytics & Visualization Helper Module
Provides Plotly chart generators, KPI calculations, and styling definitions
for the AI Insurance Cost & Risk Analytics Dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

# Sleek modern dark/light palette
THEME_COLORS = {
    'primary': '#0284c7',       # Ocean blue
    'secondary': '#6366f1',     # Indigo
    'accent': '#06b6d4',        # Cyan
    'success': '#10b981',       # Emerald
    'warning': '#f59e0b',       # Amber
    'danger': '#ef4444',        # Crimson
    'background': '#0f172a',    # Slate 900
    'surface': '#1e293b',       # Slate 800
    'text': '#f8fafc',          # Slate 50
    'muted': '#94a3b8'          # Slate 400
}

PALETTE = ['#0284c7', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4']

def apply_chart_style(fig, title: str = "", height: int = 400):
    """Apply uniform premium styling to Plotly figures."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>" if title else "",
            'font': {'size': 16, 'color': '#1e293b'}
        },
        template='plotly_white',
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        font=dict(family="Inter, Roboto, sans-serif", size=12, color="#334155"),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_color="#ffffff",
            font_family="Inter, sans-serif"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
    return fig

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI summary metrics."""
    if df.empty:
        return {
            'total_customers': 0,
            'avg_cost': 0.0,
            'median_cost': 0.0,
            'min_cost': 0.0,
            'max_cost': 0.0,
            'avg_age': 0.0,
            'avg_bmi': 0.0,
            'high_risk_pct': 0.0,
            'smoker_pct': 0.0,
            'low_risk_pct': 0.0,
            'medium_risk_pct': 0.0
        }

    total = len(df)
    avg_cost = df['charges'].mean() if 'charges' in df.columns else 0.0
    median_cost = df['charges'].median() if 'charges' in df.columns else 0.0
    min_cost = df['charges'].min() if 'charges' in df.columns else 0.0
    max_cost = df['charges'].max() if 'charges' in df.columns else 0.0
    avg_age = df['age'].mean() if 'age' in df.columns else 0.0
    avg_bmi = df['bmi'].mean() if 'bmi' in df.columns else 0.0

    smoker_count = len(df[df['smoker'].astype(str).str.lower().str.strip() == 'yes']) if 'smoker' in df.columns else 0
    smoker_pct = (smoker_count / total * 100.0) if total > 0 else 0.0

    high_risk_count = len(df[df['risk_category'] == 'HIGH']) if 'risk_category' in df.columns else 0
    high_risk_pct = (high_risk_count / total * 100.0) if total > 0 else 0.0

    medium_risk_count = len(df[df['risk_category'] == 'MEDIUM']) if 'risk_category' in df.columns else 0
    medium_risk_pct = (medium_risk_count / total * 100.0) if total > 0 else 0.0

    low_risk_count = len(df[df['risk_category'] == 'LOW']) if 'risk_category' in df.columns else 0
    low_risk_pct = (low_risk_count / total * 100.0) if total > 0 else 0.0

    return {
        'total_customers': total,
        'avg_cost': round(float(avg_cost), 2),
        'median_cost': round(float(median_cost), 2),
        'min_cost': round(float(min_cost), 2),
        'max_cost': round(float(max_cost), 2),
        'avg_age': round(float(avg_age), 1),
        'avg_bmi': round(float(avg_bmi), 1),
        'high_risk_pct': round(float(high_risk_pct), 1),
        'smoker_pct': round(float(smoker_pct), 1),
        'low_risk_pct': round(float(low_risk_pct), 1),
        'medium_risk_pct': round(float(medium_risk_pct), 1)
    }

def create_cost_distribution_chart(df: pd.DataFrame):
    """Generate cost distribution histogram with kernel density estimation."""
    fig = px.histogram(
        df,
        x='charges',
        nbins=45,
        color_discrete_sequence=['#0284c7'],
        opacity=0.85,
        marginal='box'
    )
    mean_val = df['charges'].mean()
    median_val = df['charges'].median()

    fig.add_vline(x=mean_val, line_dash='dash', line_color='#ef4444', annotation_text=f"Mean: ${mean_val:,.0f}")
    fig.add_vline(x=median_val, line_dash='dot', line_color='#10b981', annotation_text=f"Median: ${median_val:,.0f}")

    fig.update_xaxes(title="Annual Charges ($)")
    fig.update_yaxes(title="Customer Count")
    return apply_chart_style(fig, "Insurance Cost Distribution")

def create_cost_by_age_group_chart(df: pd.DataFrame):
    """Bar chart of average insurance cost by age demographic tier."""
    agg = df.groupby('age_group')['charges'].agg(['mean', 'median', 'count']).reset_index()
    fig = px.bar(
        agg,
        x='age_group',
        y='mean',
        text_auto='.2s',
        color='mean',
        color_continuous_scale='Blues',
        labels={'age_group': 'Age Group', 'mean': 'Avg Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    return apply_chart_style(fig, "Average Insurance Cost by Age Group")

def create_cost_by_bmi_category_chart(df: pd.DataFrame):
    """Bar chart of average insurance cost categorized by BMI classifications."""
    order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    agg = df.groupby('bmi_category')['charges'].mean().reindex(order).dropna().reset_index()
    fig = px.bar(
        agg,
        x='bmi_category',
        y='charges',
        color='bmi_category',
        color_discrete_sequence=PALETTE,
        labels={'bmi_category': 'BMI Category', 'charges': 'Average Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    return apply_chart_style(fig, "Average Cost by BMI Category")

def create_cost_by_smoker_chart(df: pd.DataFrame):
    """Box plot contrasting charges for smokers vs non-smokers."""
    fig = px.box(
        df,
        x='smoker',
        y='charges',
        color='smoker',
        color_discrete_map={'yes': '#ef4444', 'no': '#10b981'},
        labels={'smoker': 'Smoking Status', 'charges': 'Annual Charges ($)'},
        points='outliers'
    )
    return apply_chart_style(fig, "Insurance Cost by Smoking Status")

def create_cost_by_gender_chart(df: pd.DataFrame):
    """Bar chart comparing insurance cost across gender categories."""
    agg = df.groupby('sex')['charges'].mean().reset_index()
    fig = px.bar(
        agg,
        x='sex',
        y='charges',
        color='sex',
        color_discrete_sequence=['#8b5cf6', '#0284c7'],
        labels={'sex': 'Gender', 'charges': 'Average Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    return apply_chart_style(fig, "Average Cost by Gender")

def create_cost_by_region_chart(df: pd.DataFrame):
    """Bar chart comparing average cost by geographical territory."""
    agg = df.groupby('region')['charges'].mean().reset_index().sort_values(by='charges', ascending=False)
    fig = px.bar(
        agg,
        x='region',
        y='charges',
        color='region',
        color_discrete_sequence=PALETTE,
        labels={'region': 'Geographic Region', 'charges': 'Average Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    return apply_chart_style(fig, "Average Cost by Geographic Region")

def create_children_vs_cost_chart(df: pd.DataFrame):
    """Box / Bar chart showing relationship between dependent count and insurance cost."""
    agg = df.groupby('children')['charges'].mean().reset_index()
    fig = px.bar(
        agg,
        x='children',
        y='charges',
        color='charges',
        color_continuous_scale='Teal',
        labels={'children': 'Number of Children / Dependents', 'charges': 'Average Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    return apply_chart_style(fig, "Dependents (Children) vs Insurance Cost")

def create_age_vs_cost_scatter(df: pd.DataFrame):
    """Interactive scatter plot mapping age against charges stratified by smoking."""
    sample_df = df.sample(min(1200, len(df)), random_state=42) if len(df) > 1200 else df
    fig = px.scatter(
        sample_df,
        x='age',
        y='charges',
        color='smoker',
        color_discrete_map={'yes': '#ef4444', 'no': '#0284c7'},
        hover_data=['bmi', 'region', 'children'],
        labels={'age': 'Age (Years)', 'charges': 'Annual Charges ($)', 'smoker': 'Smoker'}
    )
    return apply_chart_style(fig, "Age vs Insurance Cost (Stratified by Smoking)")

def create_bmi_vs_cost_scatter(df: pd.DataFrame):
    """Interactive scatter plot of BMI against charges with smoker breakdown."""
    sample_df = df.sample(min(1200, len(df)), random_state=42) if len(df) > 1200 else df
    fig = px.scatter(
        sample_df,
        x='bmi',
        y='charges',
        color='smoker',
        color_discrete_map={'yes': '#ef4444', 'no': '#10b981'},
        hover_data=['age', 'region', 'children'],
        labels={'bmi': 'Body Mass Index (BMI)', 'charges': 'Annual Charges ($)', 'smoker': 'Smoker'}
    )
    fig.add_vline(x=30, line_dash='dash', line_color='#f59e0b', annotation_text="Obesity Threshold (BMI ≥ 30)")
    return apply_chart_style(fig, "BMI vs Insurance Cost (Threshold Breakdown)")

def create_risk_distribution_chart(df: pd.DataFrame):
    """Distribution of charges segmented by risk category."""
    if 'risk_category' not in df.columns:
        return go.Figure()
    fig = px.box(
        df,
        x='risk_category',
        y='charges',
        color='risk_category',
        category_orders={'risk_category': ['LOW', 'MEDIUM', 'HIGH']},
        color_discrete_map={'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'},
        labels={'risk_category': 'Actuarial Risk Category', 'charges': 'Annual Charges ($)'}
    )
    return apply_chart_style(fig, "Cost Distribution by Risk Category")

def create_risk_gauge_chart(risk_score: float, risk_category: str):
    """
    Render a calibrated Plotly Gauge Chart for risk indicator (0 to 100).
    Documented bands:
    - 0 to 35: LOW (Green)
    - 35 to 70: MEDIUM (Amber)
    - 70 to 100: HIGH (Crimson)
    """
    category_color = {
        'LOW': '#10b981',
        'MEDIUM': '#f59e0b',
        'HIGH': '#ef4444'
    }.get(risk_category, '#0284c7')

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Risk Score: {risk_score} / 100</b><br><span style='font-size:14px; color:{category_color}'>{risk_category} RISK TIER</span>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': category_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 35], 'color': '#d1fae5'},   # Soft green
                {'range': [35, 70], 'color': '#fef3c7'},  # Soft amber
                {'range': [70, 100], 'color': '#fee2e2'}  # Soft crimson
            ],
            'threshold': {
                'line': {'color': "#0f172a", 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=60, b=20),
        font=dict(family="Inter, Roboto, sans-serif")
    )
    return fig

def create_actual_vs_predicted_chart(y_true, y_pred, model_name: str):
    """Actual vs Predicted values scatter plot with ideal equality line."""
    df_plot = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(
        df_plot,
        x='Actual',
        y='Predicted',
        opacity=0.6,
        color_discrete_sequence=['#0284c7'],
        labels={'Actual': 'Actual Charges ($)', 'Predicted': 'Predicted Charges ($)'}
    )
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color='#ef4444', dash='dash', width=2)
    )
    return apply_chart_style(fig, f"Actual vs Predicted Charges ({model_name})")

def create_residual_plot(y_pred, residuals, model_name: str):
    """Residual plot (Predicted vs Residuals) to evaluate homoscedasticity."""
    df_plot = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals})
    fig = px.scatter(
        df_plot,
        x='Predicted',
        y='Residuals',
        opacity=0.6,
        color_discrete_sequence=['#8b5cf6'],
        labels={'Predicted': 'Predicted Charges ($)', 'Residuals': 'Residual (Actual - Predicted) ($)'}
    )
    fig.add_hline(y=0, line_dash='dash', line_color='#ef4444')
    return apply_chart_style(fig, f"Residuals vs Predicted ({model_name})")

def create_error_distribution_chart(residuals, model_name: str):
    """Histogram of prediction errors / residuals."""
    df_plot = pd.DataFrame({'Residuals': residuals})
    fig = px.histogram(
        df_plot,
        x='Residuals',
        nbins=40,
        marginal='violin',
        color_discrete_sequence=['#06b6d4'],
        labels={'Residuals': 'Error / Residual ($)'}
    )
    fig.add_vline(x=0, line_dash='dash', line_color='#ef4444')
    return apply_chart_style(fig, f"Error Distribution ({model_name})")

def create_feature_importance_chart(feature_importances: dict, title: str = "Feature Importance"):
    """Horizontal bar chart of model feature importances."""
    df_imp = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        text_auto='.3f'
    )
    fig.update_traces(textposition='outside')
    return apply_chart_style(fig, title, height=420)

def create_scenario_comparison_chart(scenario_df: pd.DataFrame):
    """Bar chart comparing projected costs across what-if scenarios."""
    fig = px.bar(
        scenario_df,
        x='Scenario',
        y='Predicted Annual Cost',
        text='Predicted Annual Cost',
        color='Scenario',
        color_discrete_sequence=PALETTE,
        labels={'Predicted Annual Cost': 'Projected Annual Cost ($)'}
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    return apply_chart_style(fig, "What-If Scenario Cost Comparison", height=420)


def create_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix heatmap for continuous numerical columns."""
    num_cols = [c for c in ['age', 'bmi', 'children', 'charges', 'risk_score'] if c in df.columns]
    if len(num_cols) < 2:
        return go.Figure()

    corr = df[num_cols].corr()

    labels_map = {
        'age': 'Age',
        'bmi': 'BMI',
        'children': 'Children',
        'charges': 'Annual Charges',
        'risk_score': 'Risk Score'
    }
    clean_labels = [labels_map.get(c, c) for c in num_cols]

    fig = px.imshow(
        corr.values,
        x=clean_labels,
        y=clean_labels,
        color_continuous_scale='Blues',
        zmin=-1.0,
        zmax=1.0,
        text_auto='.2f',
        labels=dict(color="Pearson Correlation")
    )
    fig.update_traces(textfont=dict(size=12))
    return apply_chart_style(fig, "Actuarial Feature Correlation Matrix (Pearson r)", height=420)


def create_risk_tier_donut_chart(df: pd.DataFrame):
    """Donut chart illustrating customer distribution by risk tier."""
    if 'risk_category' not in df.columns or df.empty:
        return go.Figure()

    counts = df['risk_category'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH']).fillna(0).reset_index()
    counts.columns = ['Risk Tier', 'Count']

    color_map = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}
    colors = [color_map.get(tier, '#0284c7') for tier in counts['Risk Tier']]

    fig = go.Figure(data=[go.Pie(
        labels=counts['Risk Tier'],
        values=counts['Count'],
        hole=0.55,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    return apply_chart_style(fig, "Customer Distribution by Risk Tier", height=380)


def create_demographic_distribution_chart(df: pd.DataFrame):
    """Grouped bar chart showing customer population across age brackets and gender."""
    if 'age_group' not in df.columns or 'sex' not in df.columns or df.empty:
        return go.Figure()

    agg = df.groupby(['age_group', 'sex']).size().reset_index()
    agg.columns = ['age_group', 'sex', 'Count']
    fig = px.bar(
        agg,
        x='age_group',
        y='Count',
        color='sex',
        barmode='group',
        color_discrete_sequence=['#8b5cf6', '#0284c7'],
        labels={'age_group': 'Age Group', 'Count': 'Customer Count', 'sex': 'Gender'}
    )
    fig.update_traces(textposition='outside')
    return apply_chart_style(fig, "Demographic Distribution by Age Group & Gender", height=380)

