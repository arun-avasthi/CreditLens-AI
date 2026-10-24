"""
CreditLens AI - Central Configuration
=====================================
All project constants, model parameters, and path configurations.
"""
import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'generated')

# ==============================================================================
# Random State & Split
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==============================================================================
# Classification Threshold
# ==============================================================================
THRESHOLD = 0.5  # Fixed project threshold — NOT an industry standard

# ==============================================================================
# XGBoost Configuration
# ==============================================================================
XGB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.05,
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'use_label_encoder': False,
}

# ==============================================================================
# MLP Configuration
# ==============================================================================
MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'early_stopping': True,
    'max_iter': 500,
    'random_state': RANDOM_STATE,
}

# ==============================================================================
# Feature Groups (UCI German Credit Dataset)
# ==============================================================================
NUMERIC_FEATURES = [
    'duration',
    'credit_amount',
    'installment_rate',
    'residence_since',
    'age',
    'existing_credits',
    'num_dependents',
]

CATEGORICAL_FEATURES = [
    'checking_account',
    'credit_history',
    'purpose',
    'savings_account',
    'employment_since',
    'personal_status_sex',
    'other_debtors',
    'property',
    'other_installment_plans',
    'housing',
    'job',
    'telephone',
    'foreign_worker',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ==============================================================================
# Artifact File Names
# ==============================================================================
ARTIFACT_FILES = {
    'xgb_model': 'xgb_model.joblib',
    'mlp_model': 'mlp_model.joblib',
    'preprocessor': 'preprocessor.joblib',
    'feature_names': 'feature_names.joblib',
    'background_samples': 'background_samples.joblib',
    'test_samples': 'test_samples.joblib',
    'metrics': 'metrics.json',
    'manifest': 'model_manifest.json',
}

# ==============================================================================
# SHAP / LIME Configuration
# ==============================================================================
LIME_NUM_SAMPLES = 1200
LIME_NUM_FEATURES = 10
SHAP_BACKGROUND_SIZE = 100

# ==============================================================================
# Human-Readable Feature Labels (for UI display)
# ==============================================================================
FEATURE_LABELS = {
    'duration': 'Loan Duration (months)',
    'credit_amount': 'Credit Amount (DM)',
    'installment_rate': 'Installment Rate (% of income)',
    'residence_since': 'Years at Current Residence',
    'age': 'Age (years)',
    'existing_credits': 'Number of Existing Credits',
    'num_dependents': 'Number of Dependents',
    'checking_account': 'Checking Account Status',
    'credit_history': 'Credit History',
    'purpose': 'Loan Purpose',
    'savings_account': 'Savings Account',
    'employment_since': 'Employment Duration',
    'personal_status_sex': 'Personal Status & Sex',
    'other_debtors': 'Other Debtors / Guarantors',
    'property': 'Property',
    'other_installment_plans': 'Other Installment Plans',
    'housing': 'Housing',
    'job': 'Job Type',
    'telephone': 'Telephone',
    'foreign_worker': 'Foreign Worker',
}
