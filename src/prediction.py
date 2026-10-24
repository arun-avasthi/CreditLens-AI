"""
CreditLens AI - Prediction Engine
===================================
Loads saved model artifacts and makes predictions on new applicant data.
Threshold: 50% (fixed project threshold).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, THRESHOLD, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_model(model_name='xgb_model'):
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return joblib.load(path)


def load_preprocessor():
    """Load the fitted preprocessor and feature names."""
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.joblib')
    
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor not found: {prep_path}. Run train.py first.")
    
    preprocessor = joblib.load(prep_path)
    feature_names = joblib.load(feat_path) if os.path.exists(feat_path) else None
    return preprocessor, feature_names


def predict(applicant_dict, model_name='xgb_model', model=None, preprocessor=None):
    """
    Make a prediction for a single applicant.
    
    Args:
        applicant_dict: dict with all 20 feature values
        model_name: 'xgb_model' or 'mlp_model'
        model: optional pre-loaded model (for efficiency)
        preprocessor: optional pre-loaded preprocessor
        
    Returns:
        dict with:
            - approval_probability: float [0, 1]
            - default_risk: float [0, 1]
            - risk_class: 'Low Risk' / 'Medium Risk' / 'High Risk'
            - approved: bool
            - threshold: float
            - model_name: str
    """
    if model is None:
        model = load_model(model_name)
    if preprocessor is None:
        preprocessor, _ = load_preprocessor()
    
    # Build DataFrame with correct column order
    df = pd.DataFrame([applicant_dict])
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")
    
    df = df[expected_cols]
    
    # Ensure types
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    
    # Transform
    X_processed = preprocessor.transform(df)
    
    # Predict
    proba = model.predict_proba(X_processed)[0]
    
    # proba[1] = P(good credit / approval)
    # proba[0] = P(bad credit / default)
    approval_prob = float(proba[1])
    default_risk = float(proba[0])
    
    # Risk classification based on threshold
    approved = approval_prob >= THRESHOLD
    
    # Risk class labels
    if default_risk < 0.3:
        risk_class = 'Low Risk'
    elif default_risk < 0.5:
        risk_class = 'Medium Risk'
    else:
        risk_class = 'High Risk'
    
    return {
        'approval_probability': approval_prob,
        'default_risk': default_risk,
        'risk_class': risk_class,
        'approved': approved,
        'threshold': THRESHOLD,
        'model_name': model_name.replace('_model', '').upper(),
    }


def predict_batch(applicant_dicts, model_name='xgb_model'):
    """Predict for multiple applicants efficiently."""
    model = load_model(model_name)
    preprocessor, _ = load_preprocessor()
    
    results = []
    for applicant in applicant_dicts:
        result = predict(applicant, model_name, model=model, preprocessor=preprocessor)
        results.append(result)
    return results
