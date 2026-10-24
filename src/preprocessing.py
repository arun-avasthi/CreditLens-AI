"""
CreditLens AI - Preprocessing Module
======================================
sklearn ColumnTransformer pipeline:
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  
Fit ONLY on training data. Transform test/new data separately.
Expected output: ~61 transformed features.
"""
import os
import sys
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR


def build_preprocessor():
    """
    Build a ColumnTransformer with StandardScaler (numeric) + OneHotEncoder (categorical).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder='drop',
    )
    return preprocessor


def save_preprocessor(preprocessor, feature_names):
    """Save fitted preprocessor and feature names to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    joblib.dump(list(feature_names), os.path.join(MODELS_DIR, 'feature_names.joblib'))


def load_preprocessor():
    """Load preprocessor and feature names from disk."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    return preprocessor, feature_names


def get_feature_names(preprocessor):
    """Extract feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                names.extend(columns)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    names.extend(transformer.get_feature_names_out(columns))
                else:
                    names.extend(transformer.get_feature_names(columns))
        return names


def preprocess_applicant(applicant_dict, preprocessor=None):
    """
    Preprocess a single applicant dict using the saved preprocessor.
    Returns numpy array of transformed features.
    """
    import pandas as pd
    
    if preprocessor is None:
        preprocessor, _ = load_preprocessor()
    
    df = pd.DataFrame([applicant_dict])
    
    # Ensure column order matches training
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")
    
    df = df[expected_cols]
    
    # Ensure numeric types
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure categorical types
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    
    X_processed = preprocessor.transform(df)
    return X_processed
