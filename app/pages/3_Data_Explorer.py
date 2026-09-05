import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer
from app.utils.data_helper import load_raw_data

st.set_page_config(page_title="Data Explorer - EMIPredict AI", page_icon="🔍", layout="wide")
apply_fintech_style()

render_header(
    title="Financial Dataset Explorer",
    subtitle="Interactive inspection, filtering, and statistical analysis across 404,800 records.",
    badge="EDA Dashboard"
)

# Load data (cached sample of 50k for ultra-responsive UI)
with st.spinner("Loading dataset cache..."):
    try:
        df_sample = load_raw_data(sample_n=50000)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Overview metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Population Scale", "404,800 Records")
m2.metric("Total Schema Attributes", f"{df_sample.shape[1]} Columns")
m3.metric("Memory In-Use", f"~{df_sample.memory_usage().sum() / (1024*1024):.1f} MB (Sample)")
m4.metric("Active Filter Sample", f"{len(df_sample):,} Records")

st.markdown("---")

# Filter controls
st.subheader("🎯 Interactive Cohort Filters")
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    selected_scenario = st.multiselect(
        "Filter by EMI Scenario:",
        options=sorted(df_sample['emi_scenario'].dropna().unique()),
        default=[]
    )

with f_col2:
    selected_eligibility = st.multiselect(
        "Filter by Eligibility Target:",
        options=sorted(df_sample['emi_eligibility'].dropna().unique()),
        default=[]
    )

with f_col3:
    salary_range = st.slider(
        "Monthly Salary Range (₹):",
        min_value=5000,
        max_value=300000,
        value=(10000, 150000),
        step=5000
    )

# Apply filters
filtered_df = df_sample.copy()
if selected_scenario:
    filtered_df = filtered_df[filtered_df['emi_scenario'].isin(selected_scenario)]
if selected_eligibility:
    filtered_df = filtered_df[filtered_df['emi_eligibility'].isin(selected_eligibility)]

# Convert salary to numeric for filtering
salary_num = pd.to_numeric(filtered_df['monthly_salary'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True), errors='coerce')
filtered_df = filtered_df[(salary_num >= salary_range[0]) & (salary_num <= salary_range[1])]

st.info(f"Showing **{len(filtered_df):,}** matching applicant records after filtering.")

# Tabs: Data View, Statistical Summary, Visualizations
tab_data, tab_stats, tab_charts = st.tabs(["📋 Data Preview", "📊 Summary Statistics", "📈 Cohort Visualizations"])

with tab_data:
    st.dataframe(filtered_df.head(100), use_container_width=True)

with tab_stats:
    st.markdown("#### Numerical Attributes Summary")
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns
    st.dataframe(filtered_df[num_cols].describe().T, use_container_width=True)
    
    st.markdown("#### Missing Values Breakdown")
    missing = pd.DataFrame({
        "Attribute": filtered_df.columns,
        "Missing Values": filtered_df.isnull().sum().values,
        "Percentage (%)": (filtered_df.isnull().sum().values / len(filtered_df) * 100).round(2)
    })
    st.dataframe(missing[missing['Missing Values'] > 0], use_container_width=True)

with tab_charts:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Distribution across EMI Scenarios")
        st.bar_chart(filtered_df['emi_scenario'].value_counts())
        
    with c2:
        st.markdown("##### Eligibility Breakdown")
        st.bar_chart(filtered_df['emi_eligibility'].value_counts(), color="#10b981")

render_disclaimer()
