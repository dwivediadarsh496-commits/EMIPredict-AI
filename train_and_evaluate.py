"""
EMIPredict AI - Model Training, Evaluation, and MLflow Experiment Tracking Pipeline
"""

import os
import sys
import time
import json
import warnings
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings('ignore')

# Set style for professional fintech charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

FIGURES_DIR = r"d:\EMIPredict AI\reports\figures"
MODELS_DIR = r"d:\EMIPredict AI\models"
REPORTS_DIR = r"d:\EMIPredict AI\reports"
MLFLOW_TRACKING_DIR = r"d:\EMIPredict AI\mlflow_runs"
MLFLOW_DB_PATH = r"d:\EMIPredict AI\mlflow.db"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

db_uri = f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"
mlflow.set_tracking_uri(db_uri)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ==========================================
csv_path = r"d:\EMIPredict AI\data\EMI_dataset.csv"
print("Loading dataset from:", csv_path)
t0 = time.time()
df_raw = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns in {time.time()-t0:.2f}s")

# ==========================================
# 2. DATA CLEANING
# ==========================================
def clean_dataset(df):
    data = df.copy()
    
    # 1. Clean age: handle string artifacts like '58.0.0'
    data['age'] = data['age'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    data['age'] = data['age'].fillna(data['age'].median()).clip(18, 75)
    
    # 2. Clean gender
    gender_map = {
        'Male': 'Male', 'MALE': 'Male', 'M': 'Male', 'male': 'Male',
        'Female': 'Female', 'FEMALE': 'Female', 'F': 'Female', 'female': 'Female'
    }
    data['gender'] = data['gender'].map(gender_map).fillna('Male')
    
    # 3. Clean marital_status
    data['marital_status'] = data['marital_status'].fillna('Married')
    
    # 4. Clean education
    data['education'] = data['education'].fillna('Graduate')
    
    # 5. Clean monthly_salary
    data['monthly_salary'] = data['monthly_salary'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['monthly_salary'] = pd.to_numeric(data['monthly_salary'], errors='coerce')
    data['monthly_salary'] = data['monthly_salary'].fillna(data['monthly_salary'].median()).clip(lower=5000)
    
    # 6. Clean employment_type
    data['employment_type'] = data['employment_type'].fillna('Private')
    
    # 7. Clean years_of_employment
    data['years_of_employment'] = pd.to_numeric(data['years_of_employment'], errors='coerce').fillna(5.0).clip(0, 50)
    
    # 8. Clean company_type
    data['company_type'] = data['company_type'].fillna('Mid-size')
    
    # 9. Clean house_type
    data['house_type'] = data['house_type'].fillna('Rented')
    
    # 10. Clean monthly_rent
    data['monthly_rent'] = pd.to_numeric(data['monthly_rent'], errors='coerce').fillna(0).clip(lower=0)
    
    # 11. Clean family_size & dependents
    data['family_size'] = pd.to_numeric(data['family_size'], errors='coerce').fillna(3).clip(1, 15)
    data['dependents'] = pd.to_numeric(data['dependents'], errors='coerce').fillna(1).clip(0, 10)
    
    # 12. Clean expense columns
    exp_cols = ['school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    for c in exp_cols:
        data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0).clip(lower=0)
        
    # 13. Clean existing_loans & current_emi_amount
    data['existing_loans'] = data['existing_loans'].fillna('No')
    data['current_emi_amount'] = pd.to_numeric(data['current_emi_amount'], errors='coerce').fillna(0).clip(lower=0)
    
    # 14. Clean credit_score
    data['credit_score'] = pd.to_numeric(data['credit_score'], errors='coerce')
    data['credit_score'] = data['credit_score'].fillna(data['credit_score'].median()).clip(300, 900)
    
    # 15. Clean bank_balance
    data['bank_balance'] = data['bank_balance'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['bank_balance'] = pd.to_numeric(data['bank_balance'], errors='coerce')
    data['bank_balance'] = data['bank_balance'].fillna(data['bank_balance'].median()).clip(lower=0)
    
    # 16. Clean emergency_fund
    data['emergency_fund'] = pd.to_numeric(data['emergency_fund'], errors='coerce')
    data['emergency_fund'] = data['emergency_fund'].fillna(data['emergency_fund'].median()).clip(lower=0)
    
    # 17. Clean emi_scenario
    data['emi_scenario'] = data['emi_scenario'].fillna('Personal Loan EMI')
    
    # 18. Clean requested_amount & requested_tenure
    data['requested_amount'] = pd.to_numeric(data['requested_amount'], errors='coerce').fillna(100000).clip(lower=5000)
    data['requested_tenure'] = pd.to_numeric(data['requested_tenure'], errors='coerce').fillna(24).clip(6, 120)
    
    return data

df_cleaned = clean_dataset(df_raw)
print(f"Data cleaning completed. Remaining missing values: {df_cleaned.isnull().sum().sum()}")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def add_engineered_features(df):
    data = df.copy()
    data['total_monthly_expenses'] = (
        data['school_fees'] + data['college_fees'] + 
        data['travel_expenses'] + data['groceries_utilities'] + 
        data['other_monthly_expenses'] + data['monthly_rent']
    )
    data['total_financial_obligations'] = data['total_monthly_expenses'] + data['current_emi_amount']
    data['disposable_income'] = data['monthly_salary'] - data['total_financial_obligations']
    data['expense_to_income_ratio'] = (data['total_monthly_expenses'] / (data['monthly_salary'] + 1.0)).clip(0, 5)
    data['debt_to_income_ratio'] = (data['current_emi_amount'] / (data['monthly_salary'] + 1.0)).clip(0, 3)
    data['savings_ratio'] = (data['bank_balance'] / (data['monthly_salary'] + 1.0)).clip(0, 20)
    data['emergency_fund_ratio'] = (data['emergency_fund'] / (data['total_monthly_expenses'] + 1.0)).clip(0, 30)
    
    # Financial Stability Score (0 to 100)
    norm_cibil = (data['credit_score'] - 300) / 600.0
    norm_exp = np.clip(data['years_of_employment'] / 15.0, 0, 1.0)
    norm_emergency = np.clip(data['emergency_fund'] / (data['total_monthly_expenses'] * 6 + 1.0), 0, 1.0)
    data['financial_stability_score'] = ((norm_cibil * 0.45 + norm_exp * 0.25 + norm_emergency * 0.30) * 100).round(2)
    
    return data

df_fe = add_engineered_features(df_cleaned)
print(f"Feature engineering completed. Total columns now: {df_fe.shape[1]}")

# ==========================================
# 4. GENERATE EDA VISUALIZATIONS
# ==========================================
print("Generating comprehensive EDA visualizations...")

# 1. Eligibility Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_fe, x='emi_eligibility', order=['Eligible', 'High_Risk', 'Not_Eligible'], palette=['#10b981', '#f59e0b', '#ef4444'])
plt.title("EMI Eligibility Target Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Eligibility Category")
plt.ylabel("Applicant Count")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_eligibility_dist.png"), dpi=200)
plt.close()

# 2. Scenario Distribution
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=df_fe, y='emi_scenario', palette='viridis', order=df_fe['emi_scenario'].value_counts().index)
plt.title("Loan / EMI Scenario Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Applicant Count")
plt.ylabel("EMI Scenario")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_scenario_dist.png"), dpi=200)
plt.close()

# 3. Monthly Salary & Max EMI Distribution (Sample of 20,000 for clean plotting)
sample_df = df_fe.sample(n=25000, random_state=42)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(sample_df['monthly_salary'], bins=40, kde=True, color='#3b82f6')
plt.title("Monthly Salary Distribution (Sample)", fontweight='bold')
plt.xlabel("Monthly Salary (INR)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(sample_df['max_monthly_emi'], bins=40, kde=True, color='#8b5cf6')
plt.title("Maximum Safe Monthly EMI Distribution (Sample)", fontweight='bold')
plt.xlabel("Maximum Monthly EMI (INR)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_salary_max_emi_dist.png"), dpi=200)
plt.close()

# 4. Credit Score vs Eligibility
plt.figure(figsize=(8, 5))
sns.boxplot(data=sample_df, x='emi_eligibility', y='credit_score', palette=['#ef4444', '#10b981', '#f59e0b'])
plt.title("Credit Score by EMI Eligibility Status", fontweight='bold')
plt.xlabel("EMI Eligibility")
plt.ylabel("Credit Score (CIBIL)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_credit_score_vs_eligibility.png"), dpi=200)
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(12, 9))
corr_cols = [
    'monthly_salary', 'credit_score', 'total_monthly_expenses', 'disposable_income',
    'current_emi_amount', 'bank_balance', 'emergency_fund', 'requested_amount',
    'financial_stability_score', 'max_monthly_emi'
]
corr_matrix = sample_df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Key Financial Features", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png"), dpi=200)
plt.close()

print("EDA plots successfully saved to reports/figures/.")

# ==========================================
# 5. MODEL PREPROCESSING & PIPELINE SETUP
# ==========================================
# Define Feature Columns
categorical_features = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario'
]

numerical_features = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure',
    'total_monthly_expenses', 'total_financial_obligations', 'disposable_income',
    'expense_to_income_ratio', 'debt_to_income_ratio', 'savings_ratio',
    'emergency_fund_ratio', 'financial_stability_score'
]

feature_columns = categorical_features + numerical_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Prepare Features (X) and Targets (y_class, y_reg)
X = df_fe[feature_columns]
y_class_raw = df_fe['emi_eligibility']
y_reg = df_fe['max_monthly_emi']

# Target Encoding for Classification
# Classes: 'Eligible', 'High_Risk', 'Not_Eligible'
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(y_class_raw)
class_names = list(label_encoder.classes_)
print(f"Target classes mapping: {dict(zip(range(len(class_names)), class_names))}")

# Stratified Sample for Training Speed & Memory Efficiency
# 100,000 stratified samples is plenty for convergence while keeping training rapid and stable
TRAIN_SAMPLE_SIZE = 120000
if len(df_fe) > TRAIN_SAMPLE_SIZE:
    print(f"Sampling {TRAIN_SAMPLE_SIZE:,} stratified records for efficient, high-performance training...")
    idx_sample, _ = train_test_split(
        np.arange(len(df_fe)),
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=y_class,
        random_state=42
    )
    X_sub = X.iloc[idx_sample].reset_index(drop=True)
    y_class_sub = y_class[idx_sample]
    y_reg_sub = y_reg.iloc[idx_sample].reset_index(drop=True)
else:
    X_sub, y_class_sub, y_reg_sub = X, y_class, y_reg

# 70% Train, 15% Validation, 15% Test Split
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X_sub, y_class_sub, y_reg_sub,
    test_size=0.30,
    stratify=y_class_sub,
    random_state=42
)

X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp,
    test_size=0.50,
    stratify=y_class_temp,
    random_state=42
)

print(f"Train set: {X_train.shape[0]:,} records")
print(f"Validation set: {X_val.shape[0]:,} records")
print(f"Test set: {X_test.shape[0]:,} records")

# ==========================================
# 6. CLASSIFICATION MODEL TRAINING & EVAL
# ==========================================
print("\n" + "="*50)
print("EXPERIMENT 1: CLASSIFICATION MODELS")
print("="*50)

mlflow.set_experiment("EMIPredict_Classification")

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
}

class_results = []
trained_class_pipelines = {}

for name, model in classification_models.items():
    print(f"\nTraining Classification Model: {name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    t_start = time.time()
    pipeline.fit(X_train, y_class_train)
    train_dur = time.time() - t_start
    
    # Evaluate on Validation and Test
    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)
    
    acc = accuracy_score(y_class_test, y_pred_test)
    prec = precision_score(y_class_test, y_pred_test, average='weighted', zero_division=0)
    rec = recall_score(y_class_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_class_test, y_pred_test, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_class_test, y_prob_test, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 0.0
        
    print(f"[{name}] Accuracy: {acc*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} (Trained in {train_dur:.1f}s)")
    
    class_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })
    trained_class_pipelines[name] = pipeline
    
    # MLflow Tracking
    with mlflow.start_run(run_name=f"Class_{name.replace(' ', '_')}"):
        mlflow.log_param("model_name", name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("training_duration_sec", train_dur)
        # Log confusion matrix artifact
        cm = confusion_matrix(y_class_test, y_pred_test)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix - {name}", fontweight='bold')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(FIGURES_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

df_class_comparison = pd.DataFrame(class_results)
print("\nClassification Benchmark Comparison:")
print(df_class_comparison.to_string(index=False))

# Select Best Classification Model (by F1-score & ROC-AUC)
best_class_row = df_class_comparison.sort_values(by=['F1', 'ROC-AUC', 'Accuracy'], ascending=False).iloc[0]
best_class_name = best_class_row['Model']
best_class_pipeline = trained_class_pipelines[best_class_name]
print(f"\n>>> Best Classification Model: {best_class_name} (F1={best_class_row['F1']}, Accuracy={best_class_row['Accuracy']*100:.2f}%)")

# ==========================================
# 7. REGRESSION MODEL TRAINING & EVAL
# ==========================================
print("\n" + "="*50)
print("EXPERIMENT 2: REGRESSION MODELS")
print("="*50)

mlflow.set_experiment("EMIPredict_Regression")

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

reg_results = []
trained_reg_pipelines = {}

for name, model in regression_models.items():
    print(f"\nTraining Regression Model: {name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    t_start = time.time()
    pipeline.fit(X_train, y_reg_train)
    train_dur = time.time() - t_start
    
    y_pred_reg_test = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_reg_test, y_pred_reg_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg_test))
    r2 = r2_score(y_reg_test, y_pred_reg_test)
    mape = mean_absolute_percentage_error(y_reg_test, y_pred_reg_test)
    
    print(f"[{name}] MAE: INR {mae:.2f} | RMSE: INR {rmse:.2f} | R2: {r2:.4f} | MAPE: {mape*100:.2f}%")
    
    reg_results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4)
    })
    trained_reg_pipelines[name] = pipeline
    
    # MLflow Tracking
    with mlflow.start_run(run_name=f"Reg_{name.replace(' ', '_')}"):
        mlflow.log_param("model_name", name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("training_duration_sec", train_dur)

df_reg_comparison = pd.DataFrame(reg_results)
print("\nRegression Benchmark Comparison:")
print(df_reg_comparison.to_string(index=False))

# Select Best Regression Model (by lowest RMSE and highest R²)
best_reg_row = df_reg_comparison.sort_values(by=['RMSE', 'MAE'], ascending=[True, True]).iloc[0]
best_reg_name = best_reg_row['Model']
best_reg_pipeline = trained_reg_pipelines[best_reg_name]
print(f"\n>>> Best Regression Model: {best_reg_name} (RMSE=INR {best_reg_row['RMSE']}, R2={best_reg_row['R2']})")

# Save Comparison Charts
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(data=df_class_comparison, x='Model', y='F1', palette='crest')
plt.title("Classification F1-Score Comparison", fontweight='bold')
plt.ylabel("F1 Score")
plt.xticks(rotation=15)

plt.subplot(1, 2, 2)
sns.barplot(data=df_reg_comparison, x='Model', y='RMSE', palette='rocket')
plt.title("Regression RMSE Comparison (Lower is Better)", fontweight='bold')
plt.ylabel("RMSE (INR)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_benchmark_comparison.png"), dpi=200)
plt.close()

# Save Model Comparison CSV
df_combined_comp = pd.concat([
    df_class_comparison.assign(Task="Classification"),
    df_reg_comparison.assign(Task="Regression")
], ignore_index=True)
df_combined_comp.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
print(f"Saved reports/model_comparison.csv")

# ==========================================
# 8. SAVE TRAINED PIPELINES & METADATA
# ==========================================
class_pipeline_path = os.path.join(MODELS_DIR, "classification_pipeline.pkl")
reg_pipeline_path = os.path.join(MODELS_DIR, "regression_pipeline.pkl")

joblib.dump(best_class_pipeline, class_pipeline_path)
joblib.dump(best_reg_pipeline, reg_pipeline_path)
print(f"Successfully saved {class_pipeline_path}")
print(f"Successfully saved {reg_pipeline_path}")

metadata = {
    "dataset_records": int(len(df_raw)),
    "features_count": len(feature_columns),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "feature_columns": feature_columns,
    "classification": {
        "best_model": best_class_name,
        "classes": class_names,
        "metrics": best_class_row.to_dict()
    },
    "regression": {
        "best_model": best_reg_name,
        "metrics": best_reg_row.to_dict()
    },
    "scenarios": list(df_fe['emi_scenario'].unique()),
    "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)
print("Saved models/model_metadata.json")

# ==========================================
# 9. SAMPLE PREDICTION DEMONSTRATION
# ==========================================
print("\n" + "="*50)
print("SAMPLE PREDICTION VERIFICATION")
print("="*50)

sample_applicant = pd.DataFrame([{
    'age': 32.0,
    'gender': 'Male',
    'marital_status': 'Married',
    'education': 'Graduate',
    'monthly_salary': 65000.0,
    'employment_type': 'Private',
    'years_of_employment': 6.0,
    'company_type': 'Large Indian',
    'house_type': 'Rented',
    'monthly_rent': 12000.0,
    'family_size': 3,
    'dependents': 1,
    'school_fees': 4000.0,
    'college_fees': 0.0,
    'travel_expenses': 5000.0,
    'groceries_utilities': 12000.0,
    'other_monthly_expenses': 6000.0,
    'existing_loans': 'No',
    'current_emi_amount': 0.0,
    'credit_score': 760.0,
    'bank_balance': 140000.0,
    'emergency_fund': 120000.0,
    'emi_scenario': 'Personal Loan EMI',
    'requested_amount': 250000.0,
    'requested_tenure': 24,
    'total_monthly_expenses': 39000.0,
    'total_financial_obligations': 39000.0,
    'disposable_income': 26000.0,
    'expense_to_income_ratio': 0.60,
    'debt_to_income_ratio': 0.0,
    'savings_ratio': 2.15,
    'emergency_fund_ratio': 3.07,
    'financial_stability_score': 74.5
}])

pred_class_idx = best_class_pipeline.predict(sample_applicant)[0]
pred_class_label = class_names[pred_class_idx]
pred_class_probs = best_class_pipeline.predict_proba(sample_applicant)[0]
pred_max_emi = best_reg_pipeline.predict(sample_applicant)[0]

print(f"Sample Applicant Profile: 32yo Married Male, Salary INR 65,000, CIBIL 760")
print(f"Predicted EMI Eligibility : {pred_class_label} (Confidence: {pred_class_probs[pred_class_idx]*100:.1f}%)")
print(f"Predicted Max Safe EMI    : INR {pred_max_emi:,.2f} / month")
print("="*50)
print("All training, evaluation, and saving tasks finished successfully!")
