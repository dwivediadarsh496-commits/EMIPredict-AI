import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "classification_pipeline.pkl")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "regression_pipeline.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

@st.cache_resource(show_spinner="Loading AI Risk Assessment Models...")
def load_models():
    """Loads serialized pipelines and metadata."""
    if not os.path.exists(CLASS_MODEL_PATH) or not os.path.exists(REG_MODEL_PATH):
        raise FileNotFoundError("Trained model pipelines not found. Please run training first.")
    
    class_pipeline = joblib.load(CLASS_MODEL_PATH)
    reg_pipeline = joblib.load(REG_MODEL_PATH)
    
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
    return class_pipeline, reg_pipeline, metadata

def predict_applicant(profile: dict) -> dict:
    """
    Takes a customer profile dict, computes engineered features,
    and returns eligibility and safe max EMI predictions.
    """
    class_pipeline, reg_pipeline, metadata = load_models()
    class_names = metadata.get("classification", {}).get("classes", ["Eligible", "High_Risk", "Not_Eligible"])
    
    # 1. Feature Engineering
    total_expenses = (
        profile.get('school_fees', 0.0) +
        profile.get('college_fees', 0.0) +
        profile.get('travel_expenses', 0.0) +
        profile.get('groceries_utilities', 0.0) +
        profile.get('other_monthly_expenses', 0.0) +
        profile.get('monthly_rent', 0.0)
    )
    total_obligations = total_expenses + profile.get('current_emi_amount', 0.0)
    salary = max(profile.get('monthly_salary', 50000.0), 5000.0)
    disposable_income = salary - total_obligations
    
    expense_ratio = float(np.clip(total_expenses / (salary + 1.0), 0, 5))
    debt_ratio = float(np.clip(profile.get('current_emi_amount', 0.0) / (salary + 1.0), 0, 3))
    savings_ratio = float(np.clip(profile.get('bank_balance', 0.0) / (salary + 1.0), 0, 20))
    emergency_ratio = float(np.clip(profile.get('emergency_fund', 0.0) / (total_expenses + 1.0), 0, 30))
    
    cibil = profile.get('credit_score', 700.0)
    years_exp = profile.get('years_of_employment', 5.0)
    emergency_fund = profile.get('emergency_fund', 50000.0)
    
    norm_cibil = (cibil - 300) / 600.0
    norm_exp = np.clip(years_exp / 15.0, 0, 1.0)
    norm_emergency = np.clip(emergency_fund / (total_expenses * 6 + 1.0), 0, 1.0)
    stability_score = float(np.round((norm_cibil * 0.45 + norm_exp * 0.25 + norm_emergency * 0.30) * 100, 2))
    
    # Build complete record matching feature columns
    full_profile = dict(profile)
    full_profile.update({
        'total_monthly_expenses': total_expenses,
        'total_financial_obligations': total_obligations,
        'disposable_income': disposable_income,
        'expense_to_income_ratio': expense_ratio,
        'debt_to_income_ratio': debt_ratio,
        'savings_ratio': savings_ratio,
        'emergency_fund_ratio': emergency_ratio,
        'financial_stability_score': stability_score
    })
    
    df_single = pd.DataFrame([full_profile])
    
    # Predict Classification
    pred_class_idx = class_pipeline.predict(df_single)[0]
    pred_label = class_names[pred_class_idx]
    
    pred_probs = {}
    if hasattr(class_pipeline, "predict_proba"):
        probs = class_pipeline.predict_proba(df_single)[0]
        for idx, cls_name in enumerate(class_names):
            pred_probs[cls_name] = round(float(probs[idx]), 4)
        confidence = round(float(probs[pred_class_idx]) * 100, 1)
    else:
        confidence = 90.0
        
    # Predict Regression
    pred_max_emi = max(float(reg_pipeline.predict(df_single)[0]), 500.0)
    
    # Requested EMI calculation
    req_amount = profile.get('requested_amount', 100000.0)
    req_tenure = max(int(profile.get('requested_tenure', 24)), 1)
    approx_req_monthly_emi = req_amount / req_tenure
    
    affordability = "Affordable" if pred_max_emi >= approx_req_monthly_emi else "Stretched"
    
    return {
        "eligibility": pred_label,
        "confidence": confidence,
        "probabilities": pred_probs,
        "max_safe_emi": round(pred_max_emi, 2),
        "requested_monthly_emi": round(approx_req_monthly_emi, 2),
        "affordability": affordability,
        "disposable_income": round(disposable_income, 2),
        "total_expenses": round(total_expenses, 2),
        "debt_to_income_ratio": round(debt_ratio * 100, 1),
        "financial_stability_score": stability_score
    }
