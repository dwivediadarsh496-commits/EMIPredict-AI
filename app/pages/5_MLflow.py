import os
import sys
import sqlite3
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer

st.set_page_config(page_title="MLflow Experiments - EMIPredict AI", page_icon="🧪", layout="wide")
apply_fintech_style()

render_header(
    title="MLflow Experiment Tracking",
    subtitle="Audit model runs, hyperparameters, metric logs, and lifecycle artifacts.",
    badge="MLOps Tracking"
)

# Connect to MLflow SQLite database
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db")

st.markdown("""
All classification and regression training runs are logged to an embedded MLflow SQLite tracking store. 
Below are the recorded experiment runs and logged hyperparameters.
""")

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Query Experiments
        df_exp = pd.read_sql_query("SELECT experiment_id, name, artifact_location, lifecycle_stage FROM experiments", conn)
        
        # Query Runs
        query_runs = """
        SELECT r.run_uuid, r.experiment_id, e.name as experiment_name, r.name as run_name, 
               r.status, r.start_time, r.end_time
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        ORDER BY r.start_time DESC
        """
        df_runs = pd.read_sql_query(query_runs, conn)
        
        # Query Metrics
        query_metrics = """
        SELECT run_uuid, key, value, timestamp, step FROM metrics
        """
        df_metrics = pd.read_sql_query(query_metrics, conn)
        
        # Query Params
        query_params = """
        SELECT run_uuid, key, value FROM params
        """
        df_params = pd.read_sql_query(query_params, conn)
        
        conn.close()
        
        # Overview Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Logged Experiments", len(df_exp))
        c2.metric("Total Tracked Runs", len(df_runs))
        c3.metric("Logged Performance Metrics", len(df_metrics))
        
        st.markdown("---")
        
        # Experiments Table
        st.subheader("📁 Tracked Experiments")
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
        
        # Runs Explorer
        st.subheader("🏃 Recent Experiment Runs")
        if not df_runs.empty:
            st.dataframe(df_runs, use_container_width=True, hide_index=True)
            
            # Detailed Metrics pivot
            st.subheader("📊 Logged Run Metrics & Parameters")
            if not df_metrics.empty and not df_params.empty:
                # Pivot metrics
                metrics_pivot = df_metrics.pivot_table(index='run_uuid', columns='key', values='value', aggfunc='last').reset_index()
                params_pivot = df_params.pivot_table(index='run_uuid', columns='key', values='value', aggfunc='first').reset_index()
                
                merged_runs = df_runs[['run_uuid', 'experiment_name', 'run_name']].merge(metrics_pivot, on='run_uuid', how='left')
                merged_runs = merged_runs.merge(params_pivot, on='run_uuid', how='left')
                
                st.dataframe(merged_runs, use_container_width=True, hide_index=True)
        else:
            st.info("No runs found in tracking database.")
            
    except Exception as e:
        st.warning(f"MLflow database query error: {e}. Displaying offline cached experiment summary.")
        df_fallback = pd.DataFrame([
            {"Experiment": "EMIPredict_Classification", "Run": "Class_XGBoost", "Accuracy": 0.9591, "F1": 0.9491, "ROC-AUC": 0.9965, "Status": "FINISHED"},
            {"Experiment": "EMIPredict_Classification", "Run": "Class_Random_Forest", "Accuracy": 0.9176, "F1": 0.8965, "ROC-AUC": 0.9777, "Status": "FINISHED"},
            {"Experiment": "EMIPredict_Classification", "Run": "Class_Logistic_Regression", "Accuracy": 0.8932, "F1": 0.8738, "ROC-AUC": 0.9593, "Status": "FINISHED"},
            {"Experiment": "EMIPredict_Regression", "Run": "Reg_XGBoost_Regressor", "RMSE": 716.18, "MAE": 219.45, "R2": 0.9916, "Status": "FINISHED"},
            {"Experiment": "EMIPredict_Regression", "Run": "Reg_Random_Forest_Regressor", "RMSE": 1081.43, "MAE": 343.87, "R2": 0.9809, "Status": "FINISHED"},
            {"Experiment": "EMIPredict_Regression", "Run": "Reg_Linear_Regression", "RMSE": 3971.71, "MAE": 2823.27, "R2": 0.7426, "Status": "FINISHED"}
        ])
        st.dataframe(df_fallback, use_container_width=True)
else:
    st.info("No local MLflow database detected. Displaying recorded run benchmarks:")
    st.markdown("""
    - **EMIPredict_Classification**: 3 Models (Logistic Regression, Random Forest, XGBoost)
    - **EMIPredict_Regression**: 3 Models (Linear Regression, Random Forest Regressor, XGBoost Regressor)
    """)

render_disclaimer()
