import os
import sys
import json
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer

st.set_page_config(page_title="Model Performance - EMIPredict AI", page_icon="📈", layout="wide")
apply_fintech_style()

render_header(
    title="Model Benchmarks & Performance",
    subtitle="Comparative evaluation of linear, tree ensemble, and gradient boosting algorithms.",
    badge="Model Governance"
)

# Load comparison data
comp_file = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "model_comparison.csv")
metadata_file = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model_metadata.json")

metadata = {}
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

# Highlight Best Models
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="fintech-card" style="border-left: 4px solid #10b981;">
        <div class="fintech-card-label">Selected Classification Champion</div>
        <div class="fintech-card-value">{metadata.get('classification', {}).get('best_model', 'XGBoost')}</div>
        <div class="fintech-card-sub">Test Accuracy: <strong>95.91%</strong> | Macro F1: <strong>0.9491</strong> | ROC-AUC: <strong>0.9965</strong></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="fintech-card" style="border-left: 4px solid #3b82f6;">
        <div class="fintech-card-label">Selected Regression Champion</div>
        <div class="fintech-card-value">{metadata.get('regression', {}).get('best_model', 'XGBoost Regressor')}</div>
        <div class="fintech-card-sub">Test RMSE: <strong>₹716.18</strong> | R²: <strong>0.9916</strong> | MAE: <strong>₹219.45</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab_class, tab_reg, tab_artifacts = st.tabs([
    "🎯 Task A: Classification Evaluation",
    "💵 Task B: Regression Evaluation",
    "🖼️ Evaluation Visualizations"
])

with tab_class:
    st.subheader("Classification Algorithm Benchmark")
    df_class = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": "89.32%", "Precision": 0.8605, "Recall": 0.8932, "F1-Score": 0.8738, "ROC-AUC": 0.9593, "Status": "Baseline"},
        {"Model": "Random Forest", "Accuracy": "91.76%", "Precision": 0.8771, "Recall": 0.9176, "F1-Score": 0.8965, "ROC-AUC": 0.9777, "Status": "Ensemble"},
        {"Model": "XGBoost Classifier", "Accuracy": "95.91%", "Precision": 0.9534, "Recall": 0.9591, "F1-Score": 0.9491, "ROC-AUC": 0.9965, "Status": "⭐ Selected Best"}
    ])
    st.dataframe(df_class, use_container_width=True, hide_index=True)
    
    st.markdown("#### F1-Score & Accuracy Comparison")
    df_chart = pd.DataFrame({
        "Model": ["Logistic Reg", "Random Forest", "XGBoost"],
        "Accuracy (%)": [89.32, 91.76, 95.91],
        "F1-Score (x100)": [87.38, 89.65, 94.91]
    }).set_index("Model")
    st.bar_chart(df_chart)

with tab_reg:
    st.subheader("Regression Algorithm Benchmark (Predicting Max Monthly EMI)")
    df_reg = pd.DataFrame([
        {"Model": "Linear Regression", "MAE (₹)": 2823.27, "RMSE (₹)": 3971.71, "R² Score": 0.7426, "MAPE (%)": "181.28%", "Status": "Baseline"},
        {"Model": "Random Forest Regressor", "MAE (₹)": 343.87, "RMSE (₹)": 1081.43, "R² Score": 0.9809, "MAPE (%)": "7.42%", "Status": "Strong Candidate"},
        {"Model": "XGBoost Regressor", "MAE (₹)": 219.45, "RMSE (₹)": 716.18, "R² Score": 0.9916, "MAPE (%)": "7.66%", "Status": "⭐ Selected Best"}
    ])
    st.dataframe(df_reg, use_container_width=True, hide_index=True)
    
    st.markdown("#### RMSE Error Comparison (Lower is Superior)")
    df_reg_chart = pd.DataFrame({
        "Model": ["Linear Reg", "Random Forest", "XGBoost"],
        "RMSE (₹)": [3971.71, 1081.43, 716.18]
    }).set_index("Model")
    st.bar_chart(df_reg_chart, color="#8b5cf6")

with tab_artifacts:
    st.subheader("Generated Evaluation Artifacts")
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        cm_path = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures", "cm_xgboost.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix - Best Classification Model (XGBoost)", use_container_width=True)
            
    with fig_col2:
        comp_chart = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures", "model_benchmark_comparison.png")
        if os.path.exists(comp_chart):
            st.image(comp_chart, caption="Algorithm Performance Comparison Overview", use_container_width=True)

render_disclaimer()
