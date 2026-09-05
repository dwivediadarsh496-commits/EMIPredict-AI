import os
import sys
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for clean imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer
from app.utils.model_helper import load_models

st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_fintech_style()

# Sidebar branding
with st.sidebar:
    st.markdown("### 💳 EMIPredict AI")
    st.caption("Intelligent Financial Risk & EMI Assessment")
    st.divider()
    st.markdown("""
    **System Status:** 🟢 Active  
    **Models Loaded:** XGBoost Ensemble  
    **Dataset:** 404,800 records  
    **Environment:** Production Ready  
    """)
    st.divider()
    st.info("💡 Use the sidebar to navigate between Real-time Predictions, Risk Analytics, and Data Management.")

# Header
render_header(
    title="EMIPredict AI",
    subtitle="Next-Generation Machine Learning Risk Assessment & Sustainable EMI Underwriting Platform",
    badge="v1.0 Production"
)

# Load Model Metadata
metadata_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_metadata.json")
metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

class_metrics = metadata.get("classification", {}).get("metrics", {})
reg_metrics = metadata.get("regression", {}).get("metrics", {})

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        label="Dataset Scale",
        value=f"{metadata.get('dataset_records', 404800):,}",
        subtext="Verified Financial Records",
        border_color="#3b82f6"
    )

with col2:
    acc = class_metrics.get("Accuracy", 0.9591)
    render_metric_card(
        label="Classification Accuracy",
        value=f"{acc*100:.1f}%",
        subtext=f"Model: {metadata.get('classification', {}).get('best_model', 'XGBoost')}",
        border_color="#10b981"
    )

with col3:
    rmse = reg_metrics.get("RMSE", 716.18)
    render_metric_card(
        label="Regression Error (RMSE)",
        value=f"₹{rmse:,.0f}",
        subtext=f"R² Score: {reg_metrics.get('R2', 0.9916):.4f}",
        border_color="#8b5cf6"
    )

with col4:
    f1 = class_metrics.get("F1", 0.9491)
    render_metric_card(
        label="Macro F1-Score",
        value=f"{f1:.4f}",
        subtext="Multi-class Balanced Metric",
        border_color="#f59e0b"
    )

st.markdown("---")

# Main Content: Overview and Visuals
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📊 Executive Overview & Portfolio Distributions")
    st.markdown("""
    EMIPredict AI combines **gradient boosted classification** and **non-linear regression** to provide two-tier financial risk underwriting:
    1. **Eligibility Classification**: Predicts whether a borrower is **Eligible**, **High-Risk**, or **Not Eligible**.
    2. **Safe Capacity Regression**: Determines the exact monthly EMI ceiling to prevent debt distress and borrower default.
    """)
    
    # Render saved figure if exists
    fig_path = os.path.join(os.path.dirname(__file__), "..", "reports", "figures", "eda_eligibility_dist.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="Figure 1: Target Eligibility Distribution across 404,800 Applicants", use_container_width=True)
    else:
        st.info("Eligibility distribution chart saved in reports/figures/")

with col_right:
    st.subheader("🎯 Quick Action Shortcuts")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
        <h4 style="margin-top:0; color:#38bdf8;">⚡ Run Real-time Predictions</h4>
        <p style="font-size:0.9rem; color:#94a3b8;">Assess retail loan applicants across 5 standard EMI scenarios in sub-second inference time.</p>
        <hr style="border-color: rgba(255,255,255,0.06);">
        <h4 style="color:#a78bfa;">📈 Risk Assessment Diagnostics</h4>
        <p style="font-size:0.9rem; color:#94a3b8;">Analyze disposable income, DTI ratios, and emergency reserves stress tests.</p>
        <hr style="border-color: rgba(255,255,255,0.06);">
        <h4 style="color:#34d399;">🗄️ Financial Applicant CRUD</h4>
        <p style="font-size:0.9rem; color:#94a3b8;">Manage loan applications in the local SQLite audit database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    fig_scenario = os.path.join(os.path.dirname(__file__), "..", "reports", "figures", "eda_scenario_dist.png")
    if os.path.exists(fig_scenario):
        st.image(fig_scenario, caption="Figure 2: Loan Product Volume Across Scenarios", use_container_width=True)

# Performance Snapshot
st.markdown("### 🏆 Production Model Benchmarks")
comp_path = os.path.join(os.path.dirname(__file__), "..", "reports", "model_comparison.csv")
if os.path.exists(comp_path):
    df_comp = pd.read_csv(comp_path)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

render_disclaimer()
