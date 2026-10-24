"""
CreditLens AI - Data Loading Module
====================================
Loads the UCI Statlog German Credit dataset from OpenML.

Dataset Facts:
- 1,000 records, 20 attributes (7 numeric, 13 categorical)
- Original UCI target: 1 = good credit, 2 = bad credit
- Project mapping: good → 1 (approval), bad → 0 (default)
"""
import pandas as pd
import numpy as np
import os
import sys

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# OpenML column names → our internal names
_OPENML_COLUMN_MAP = {
    'checking_status': 'checking_account',
    'duration': 'duration',
    'credit_history': 'credit_history',
    'purpose': 'purpose',
    'credit_amount': 'credit_amount',
    'savings_status': 'savings_account',
    'employment': 'employment_since',
    'installment_commitment': 'installment_rate',
    'personal_status': 'personal_status_sex',
    'other_parties': 'other_debtors',
    'residence_since': 'residence_since',
    'property_magnitude': 'property',
    'age': 'age',
    'other_payment_plans': 'other_installment_plans',
    'housing': 'housing',
    'existing_credits': 'existing_credits',
    'job': 'job',
    'num_dependents': 'num_dependents',
    'own_telephone': 'telephone',
    'foreign_worker': 'foreign_worker',
    'class': 'class',
}


def load_data():
    """
    Load the UCI German Credit dataset.
    
    Returns:
        pd.DataFrame with columns matching NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['class']
        Target: 1 = good credit (approval), 0 = bad credit (default)
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=31, as_frame=True, parser='auto')
        df = data.frame.copy()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset from OpenML: {e}\n"
            "Please ensure you have internet access for the first run."
        )
    
    # Rename columns to our internal names
    df = df.rename(columns=_OPENML_COLUMN_MAP)
    
    # Map target: 'good' → 1, 'bad' → 0
    if df['class'].dtype == object or hasattr(df['class'], 'cat'):
        df['class'] = df['class'].map({'good': 1, 'bad': 0})
    else:
        # Numeric: UCI 1=good→1, UCI 2=bad→0
        df['class'] = df['class'].map({1: 1, 2: 0})
    
    # Verify no nulls in target
    if df['class'].isnull().any():
        raise ValueError("Target column contains unexpected values after mapping.")
    
    df['class'] = df['class'].astype(int)
    
    # Ensure numeric features are numeric
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure categorical features are strings
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Select only the columns we need
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['class']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    df = df[expected_cols].copy()
    
    # Basic validation
    assert len(df) == 1000, f"Expected 1000 records, got {len(df)}"
    assert df['class'].isin([0, 1]).all(), "Target must be binary (0/1)"
    
    return df


def get_dataset_info():
    """Return summary information about the dataset."""
    return {
        'name': 'UCI Statlog German Credit',
        'records': 1000,
        'numeric_features': len(NUMERIC_FEATURES),
        'categorical_features': len(CATEGORICAL_FEATURES),
        'total_features': len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        'source': 'UCI Machine Learning Repository (OpenML ID: 31)',
        'year': '1994',
        'currency': 'Deutsche Mark (DM)',
    }
