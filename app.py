"""
CreditLens AI — Explainable Credit Risk Analysis
===================================================
Main Streamlit application.

A professional, modern dashboard for credit risk prediction and model explainability.
Uses XGBoost + MLP with SHAP and LIME explanations.

Run: streamlit run app.py
"""
import os
import sys
import json
import copy
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st

warnings.filterwarnings('ignore')

# Project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from config import (
    MODELS_DIR, THRESHOLD, RANDOM_STATE, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    FEATURE_LABELS, ALL_FEATURES
)
from src.feature_mapping import (
    FEATURE_VALUE_MAPPINGS, FEATURE_DISPLAY_NAMES,
    get_human_readable_value, get_feature_display_name, format_applicant_for_display
)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="CreditLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — Premium Fintech Dark Theme
# ==============================================================================
st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4FC3F7 0%, #00BCD4 50%, #0288D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header .subtitle {
        color: #90A4AE;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #1E2538 100%);
        border: 1px solid #2A3040;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.4rem 0;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #4FC3F7;
    }
    .metric-card h3 {
        color: #90A4AE;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Risk indicators */
    .risk-low { color: #66BB6A; }
    .risk-medium { color: #FFA726; }
    .risk-high { color: #EF5350; }

    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 24px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    .risk-badge-low { background: rgba(102,187,106,0.12); color: #66BB6A; border: 1px solid rgba(102,187,106,0.4); }
    .risk-badge-medium { background: rgba(255,167,38,0.12); color: #FFA726; border: 1px solid rgba(255,167,38,0.4); }
    .risk-badge-high { background: rgba(239,83,80,0.12); color: #EF5350; border: 1px solid rgba(239,83,80,0.4); }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E0E0E0;
        border-bottom: 2px solid #4FC3F7;
        padding-bottom: 0.4rem;
        margin: 1rem 0 0.8rem 0;
    }

    /* Info boxes */
    .info-box {
        background: rgba(79,195,247,0.06);
        border-left: 3px solid #4FC3F7;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #B0BEC5;
        line-height: 1.5;
    }
    .warning-box {
        background: rgba(255,167,38,0.06);
        border-left: 3px solid #FFA726;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #FFCC80;
        line-height: 1.5;
    }
    .success-box {
        background: rgba(102,187,106,0.06);
        border-left: 3px solid #66BB6A;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #A5D6A7;
        line-height: 1.5;
    }

    /* Workflow steps */
    .workflow {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .workflow-step {
        background: linear-gradient(135deg, #1A1F2E, #1E2538);
        border: 1px solid #2A3040;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-size: 0.82rem;
        color: #B0BEC5;
        text-align: center;
    }
    .workflow-arrow { color: #4FC3F7; font-size: 1.1rem; }

    /* Probability bar */
    .prob-bar-container {
        background: #1A1F2E;
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        margin: 0.3rem 0;
        border: 1px solid #2A3040;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Clean up metric display */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #1E2538 100%);
        border: 1px solid #2A3040;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }

    /* Stat grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    .stat-item {
        background: linear-gradient(135deg, #1A1F2E 0%, #1E2538 100%);
        border: 1px solid #2A3040;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-item .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4FC3F7;
    }
    .stat-item .stat-label {
        font-size: 0.72rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* Logo area */
    .logo-area {
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .logo-area .logo-text {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4FC3F7 0%, #00BCD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .logo-area .logo-sub {
        font-size: 0.65rem;
        color: #607D8B;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Caching & Model Loading
# ==============================================================================
@st.cache_resource
def load_models():
    """Load trained models from disk."""
    import joblib
    models = {}
    for name in ['xgb_model', 'mlp_model']:
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_preprocessor_cached():
    """Load preprocessor and feature names."""
    import joblib
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.joblib')
    if os.path.exists(prep_path):
        preprocessor = joblib.load(prep_path)
        feature_names = joblib.load(feat_path) if os.path.exists(feat_path) else []
        return preprocessor, list(feature_names)
    return None, []

@st.cache_resource
def load_background_samples():
    """Load background samples for SHAP."""
    import joblib
    path = os.path.join(MODELS_DIR, 'background_samples.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_training_samples():
    """Load training samples for LIME."""
    import joblib
    path = os.path.join(MODELS_DIR, 'test_samples.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_metrics_cached():
    """Load saved metrics."""
    path = os.path.join(MODELS_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_manifest_cached():
    """Load model manifest."""
    path = os.path.join(MODELS_DIR, 'model_manifest.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def check_artifacts():
    """Check if all required artifacts exist."""
    required = ['xgb_model.joblib', 'mlp_model.joblib', 'preprocessor.joblib', 'feature_names.joblib']
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    return len(missing) == 0, missing


# ==============================================================================
# Demo Profiles
# ==============================================================================
LOW_RISK_PROFILE = {
    'duration': 12,
    'credit_amount': 2500,
    'installment_rate': 2,
    'residence_since': 4,
    'age': 45,
    'existing_credits': 1,
    'num_dependents': 1,
    'checking_account': "'>= 200'",
    'credit_history': "'existing paid'",
    'purpose': 'radio/tv',
    'savings_account': "'500<=X<1000'",
    'employment_since': "'>=7'",
    'personal_status_sex': "'male single'",
    'other_debtors': 'none',
    'property': "'real estate'",
    'other_installment_plans': 'none',
    'housing': 'own',
    'job': 'skilled',
    'telephone': 'yes',
    'foreign_worker': 'yes',
}

HIGH_RISK_PROFILE = {
    'duration': 48,
    'credit_amount': 12000,
    'installment_rate': 4,
    'residence_since': 1,
    'age': 23,
    'existing_credits': 3,
    'num_dependents': 2,
    'checking_account': "'<0'",
    'credit_history': "'critical/other existing credit'",
    'purpose': "'new car'",
    'savings_account': "'<100'",
    'employment_since': "'<1'",
    'personal_status_sex': "'male single'",
    'other_debtors': 'none',
    'property': "'no known property'",
    'other_installment_plans': 'bank',
    'housing': 'rent',
    'job': "'unskilled resident'",
    'telephone': 'none',
    'foreign_worker': 'yes',
}

MODERATE_RISK_PROFILE = {
    'duration': 24,
    'credit_amount': 5000,
    'installment_rate': 3,
    'residence_since': 2,
    'age': 32,
    'existing_credits': 2,
    'num_dependents': 1,
    'checking_account': "'0<=X<200'",
    'credit_history': "'existing paid'",
    'purpose': 'furniture/equipment',
    'savings_account': "'100<=X<500'",
    'employment_since': "'1<=X<4'",
    'personal_status_sex': "'female div/dep/mar'",
    'other_debtors': 'none',
    'property': "'life insurance'",
    'other_installment_plans': 'none',
    'housing': 'rent',
    'job': 'skilled',
    'telephone': 'yes',
    'foreign_worker': 'yes',
}


# ==============================================================================
# Helper functions
# ==============================================================================
def get_categorical_options_for_ui(feature_name):
    """Get display options for a categorical feature dropdown."""
    mapping = FEATURE_VALUE_MAPPINGS.get(feature_name, {})
    # Filter to only real options (not A-code duplicates)
    seen_labels = {}
    for code, label in mapping.items():
        if not code.startswith('A') or code.startswith("'"):
            if label not in seen_labels.values():
                seen_labels[code] = label
    if not seen_labels:
        for code, label in mapping.items():
            if label not in seen_labels.values():
                seen_labels[code] = label
    return seen_labels


def find_code_for_label(feature_name, label):
    """Find the internal code for a human-readable label."""
    mapping = FEATURE_VALUE_MAPPINGS.get(feature_name, {})
    for code, display_label in mapping.items():
        if display_label == label:
            return code
    return label


def make_prediction(applicant_data, model_name='xgb_model'):
    """Make a prediction using cached model and preprocessor."""
    models = load_models()
    preprocessor, feature_names = load_preprocessor_cached()

    if model_name not in models:
        st.error(f"Model '{model_name}' not found. Please run training first.")
        return None

    model = models[model_name]

    df = pd.DataFrame([applicant_data])
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[expected_cols]

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    X_processed = preprocessor.transform(df)
    proba = model.predict_proba(X_processed)[0]

    approval_prob = float(proba[1])
    default_risk = float(proba[0])
    approved = approval_prob >= THRESHOLD

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
        'X_processed': X_processed,
    }


def render_probability_bar(probability, label="Approval Probability", color_scheme="auto"):
    """Render a visual probability bar."""
    pct = probability * 100
    if color_scheme == "auto":
        if probability >= 0.7:
            color = "linear-gradient(90deg, #43A047, #66BB6A)"
        elif probability >= 0.5:
            color = "linear-gradient(90deg, #F9A825, #FFA726)"
        else:
            color = "linear-gradient(90deg, #E53935, #EF5350)"
    else:
        color = color_scheme

    st.markdown(f"""
    <div style="margin: 0.3rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 0.82rem; color: #90A4AE;">{label}</span>
            <span style="font-size: 0.9rem; font-weight: 600; color: #E0E0E0;">{pct:.1f}%</span>
        </div>
        <div class="prob-bar-container">
            <div class="prob-bar-fill" style="width: {pct}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_applicant_form_inputs(defaults, key_prefix=""):
    """Render the applicant input form and return the collected data."""
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        age = st.number_input("Age", min_value=18, max_value=80, value=int(defaults['age']),
                              help="Applicant's age in years", key=f"{key_prefix}age")
    with pcol2:
        emp_options = get_categorical_options_for_ui('employment_since')
        emp_labels = list(emp_options.values())
        default_emp_label = get_human_readable_value('employment_since', defaults['employment_since'])
        emp_idx = emp_labels.index(default_emp_label) if default_emp_label in emp_labels else 0
        employment = st.selectbox("Employment Duration", emp_labels, index=emp_idx, key=f"{key_prefix}emp")
    with pcol3:
        housing_options = get_categorical_options_for_ui('housing')
        housing_labels = list(housing_options.values())
        default_housing_label = get_human_readable_value('housing', defaults['housing'])
        housing_idx = housing_labels.index(default_housing_label) if default_housing_label in housing_labels else 0
        housing = st.selectbox("Housing", housing_labels, index=housing_idx, key=f"{key_prefix}housing")

    ps_col1, ps_col2 = st.columns(2)
    with ps_col1:
        ps_options = get_categorical_options_for_ui('personal_status_sex')
        ps_labels = list(ps_options.values())
        default_ps_label = get_human_readable_value('personal_status_sex', defaults['personal_status_sex'])
        ps_idx = ps_labels.index(default_ps_label) if default_ps_label in ps_labels else 0
        personal_status = st.selectbox("Personal Status & Sex", ps_labels, index=ps_idx, key=f"{key_prefix}ps")
    with ps_col2:
        residence = st.number_input("Residence (years)", min_value=1, max_value=4,
                                    value=int(defaults['residence_since']), key=f"{key_prefix}res")

    st.markdown('<div class="section-header">💰 Financial Information</div>', unsafe_allow_html=True)
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        credit_amount = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000,
                                        value=int(defaults['credit_amount']), step=100,
                                        help="Loan amount in Deutsche Mark", key=f"{key_prefix}amt")
        installment = st.slider("Installment Rate (% of income)", 1, 4, int(defaults['installment_rate']),
                               help="Installment as percentage of disposable income", key=f"{key_prefix}inst")
        check_options = get_categorical_options_for_ui('checking_account')
        check_labels = list(check_options.values())
        default_check_label = get_human_readable_value('checking_account', defaults['checking_account'])
        check_idx = check_labels.index(default_check_label) if default_check_label in check_labels else 0
        checking = st.selectbox("Checking Account Status", check_labels, index=check_idx, key=f"{key_prefix}check")

    with fcol2:
        duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72,
                                   value=int(defaults['duration']), step=1, key=f"{key_prefix}dur")
        num_dependents = st.number_input("Dependents", min_value=1, max_value=2,
                                         value=int(defaults['num_dependents']), key=f"{key_prefix}dep")
        sav_options = get_categorical_options_for_ui('savings_account')
        sav_labels = list(sav_options.values())
        default_sav_label = get_human_readable_value('savings_account', defaults['savings_account'])
        sav_idx = sav_labels.index(default_sav_label) if default_sav_label in sav_labels else 0
        savings = st.selectbox("Savings Account", sav_labels, index=sav_idx, key=f"{key_prefix}sav")

    st.markdown('<div class="section-header">📑 Credit Information</div>', unsafe_allow_html=True)
    ccol1, ccol2 = st.columns(2)

    with ccol1:
        ch_options = get_categorical_options_for_ui('credit_history')
        ch_labels = list(ch_options.values())
        default_ch_label = get_human_readable_value('credit_history', defaults['credit_history'])
        ch_idx = ch_labels.index(default_ch_label) if default_ch_label in ch_labels else 0
        credit_history = st.selectbox("Credit History", ch_labels, index=ch_idx, key=f"{key_prefix}ch")

        existing_credits = st.number_input("Existing Credits", min_value=1, max_value=4,
                                           value=int(defaults['existing_credits']), key=f"{key_prefix}ec")

        purp_options = get_categorical_options_for_ui('purpose')
        purp_labels = list(purp_options.values())
        default_purp_label = get_human_readable_value('purpose', defaults['purpose'])
        purp_idx = purp_labels.index(default_purp_label) if default_purp_label in purp_labels else 0
        purpose = st.selectbox("Loan Purpose", purp_labels, index=purp_idx, key=f"{key_prefix}purp")

    with ccol2:
        od_options = get_categorical_options_for_ui('other_debtors')
        od_labels = list(od_options.values())
        default_od_label = get_human_readable_value('other_debtors', defaults['other_debtors'])
        od_idx = od_labels.index(default_od_label) if default_od_label in od_labels else 0
        other_debtors = st.selectbox("Other Debtors / Guarantors", od_labels, index=od_idx, key=f"{key_prefix}od")

        prop_options = get_categorical_options_for_ui('property')
        prop_labels = list(prop_options.values())
        default_prop_label = get_human_readable_value('property', defaults['property'])
        prop_idx = prop_labels.index(default_prop_label) if default_prop_label in prop_labels else 0
        property_val = st.selectbox("Property", prop_labels, index=prop_idx, key=f"{key_prefix}prop")

        oip_options = get_categorical_options_for_ui('other_installment_plans')
        oip_labels = list(oip_options.values())
        default_oip_label = get_human_readable_value('other_installment_plans', defaults['other_installment_plans'])
        oip_idx = oip_labels.index(default_oip_label) if default_oip_label in oip_labels else 0
        other_plans = st.selectbox("Other Installment Plans", oip_labels, index=oip_idx, key=f"{key_prefix}oip")

    with st.expander("📋 Additional Details"):
        add_col1, add_col2, add_col3 = st.columns(3)
        with add_col1:
            job_options = get_categorical_options_for_ui('job')
            job_labels = list(job_options.values())
            default_job_label = get_human_readable_value('job', defaults['job'])
            job_idx = job_labels.index(default_job_label) if default_job_label in job_labels else 0
            job = st.selectbox("Job Type", job_labels, index=job_idx, key=f"{key_prefix}job")
        with add_col2:
            tel_options = get_categorical_options_for_ui('telephone')
            tel_labels = list(tel_options.values())
            default_tel_label = get_human_readable_value('telephone', defaults['telephone'])
            tel_idx = tel_labels.index(default_tel_label) if default_tel_label in tel_labels else 0
            telephone = st.selectbox("Telephone", tel_labels, index=tel_idx, key=f"{key_prefix}tel")
        with add_col3:
            fw_options = get_categorical_options_for_ui('foreign_worker')
            fw_labels = list(fw_options.values())
            default_fw_label = get_human_readable_value('foreign_worker', defaults['foreign_worker'])
            fw_idx = fw_labels.index(default_fw_label) if default_fw_label in fw_labels else 0
            foreign_worker = st.selectbox("Foreign Worker", fw_labels, index=fw_idx, key=f"{key_prefix}fw")

    # Build applicant dict with internal codes
    applicant = {
        'duration': duration,
        'credit_amount': credit_amount,
        'installment_rate': installment,
        'residence_since': residence,
        'age': age,
        'existing_credits': existing_credits,
        'num_dependents': num_dependents,
        'checking_account': find_code_for_label('checking_account', checking),
        'credit_history': find_code_for_label('credit_history', credit_history),
        'purpose': find_code_for_label('purpose', purpose),
        'savings_account': find_code_for_label('savings_account', savings),
        'employment_since': find_code_for_label('employment_since', employment),
        'personal_status_sex': find_code_for_label('personal_status_sex', personal_status),
        'other_debtors': find_code_for_label('other_debtors', other_debtors),
        'property': find_code_for_label('property', property_val),
        'other_installment_plans': find_code_for_label('other_installment_plans', other_plans),
        'housing': find_code_for_label('housing', housing),
        'job': find_code_for_label('job', job),
        'telephone': find_code_for_label('telephone', telephone),
        'foreign_worker': find_code_for_label('foreign_worker', foreign_worker),
    }
    return applicant


def compute_shap_for_current():
    """Compute SHAP values for the current prediction if not already done."""
    result = st.session_state.get('prediction_result')
    model_name = st.session_state.get('model_name', 'xgb_model')
    if result is None or model_name != 'xgb_model':
        return None, None, None

    from src.explainability import get_shap_explainer, compute_shap_values
    models = load_models()
    _, feature_names = load_preprocessor_cached()
    bg_samples = load_background_samples()

    xgb = models.get('xgb_model')
    if xgb is None:
        return None, None, None

    explainer = get_shap_explainer(xgb, bg_samples)
    shap_vals, base_val = compute_shap_values(explainer, result['X_processed'])
    return shap_vals, base_val, feature_names


# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-text">🔍 CreditLens AI</div>
        <div class="logo-sub">Explainable Credit Risk</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "👤 Credit Risk Prediction",
            "🌊 SHAP Explainability",
            "🍋 LIME Explainability",
            "⚖️ SHAP vs LIME",
            "🧪 What-if Analysis",
            "⚠️ Stress Testing",
            "🌍 Global Feature Importance",
            "📊 Model Performance",
            "🛡️ Fairness & Responsible AI",
            "📄 Reports & Downloads",
            "ℹ️ About the Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color: #607D8B; font-size: 0.72rem; text-align: center;">'
        'Educational Prototype<br>Not for real lending decisions</p>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# Check artifacts
# ==============================================================================
artifacts_ok, missing_artifacts = check_artifacts()
if not artifacts_ok:
    st.error(f"⚠️ Missing model artifacts: {', '.join(missing_artifacts)}")
    st.info("Run `python train.py` to generate model artifacts before using the dashboard.")
    st.stop()


# ==============================================================================
# PAGE: Overview Dashboard
# ==============================================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>CreditLens AI</h1>
        <p class="subtitle">Explainable Credit Risk Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        An educational machine-learning prototype that predicts credit risk and explains
        model decisions using <strong>XGBoost</strong>, <strong>MLP</strong>,
        <strong>SHAP</strong>, and <strong>LIME</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Workflow
    st.markdown("""
    <div class="workflow">
        <div class="workflow-step">📝 Applicant Data</div>
        <span class="workflow-arrow">→</span>
        <div class="workflow-step">⚙️ Prediction</div>
        <span class="workflow-arrow">→</span>
        <div class="workflow-step">💡 Explanation</div>
        <span class="workflow-arrow">→</span>
        <div class="workflow-step">📊 Analysis</div>
        <span class="workflow-arrow">→</span>
        <div class="workflow-step">📄 Report</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    metrics_data = load_metrics_cached()
    manifest = load_manifest_cached()

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", "2", help="XGBoost + MLP")
    col2.metric("Dataset", "1,000 records", help="UCI German Credit")
    col3.metric("Features", f"{manifest.get('n_features', 61)} transformed" if manifest else "61 transformed")
    col4.metric("Train / Test", f"{manifest.get('train_size', 800)} / {manifest.get('test_size', 200)}" if manifest else "800 / 200")

    st.markdown("---")

    if metrics_data:
        st.markdown("##### 📈 Model Performance (Held-out Test Set)")
        xgb_m = metrics_data.get('xgb', {}).get('metrics', {})
        mlp_m = metrics_data.get('mlp', {}).get('metrics', {})

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        perf_col1.metric("XGBoost ROC-AUC", f"{xgb_m.get('roc_auc', 0):.3f}")
        perf_col2.metric("XGBoost F1", f"{xgb_m.get('f1', 0):.3f}")
        perf_col3.metric("MLP ROC-AUC", f"{mlp_m.get('roc_auc', 0):.3f}")
        perf_col4.metric("MLP F1", f"{mlp_m.get('f1', 0):.3f}")

    st.markdown("---")

    # Key capabilities
    st.markdown("##### 🎯 Key Capabilities")
    cap_col1, cap_col2, cap_col3 = st.columns(3)
    with cap_col1:
        st.markdown("**🤖 Dual Models**")
        st.markdown("Compare **XGBoost** (gradient boosting) with **MLP** (neural network) predictions side by side.")
    with cap_col2:
        st.markdown("**💡 Explainability**")
        st.markdown("Understand **why** the model makes each prediction using **SHAP** and **LIME** explanations.")
    with cap_col3:
        st.markdown("**📊 Analysis Tools**")
        st.markdown("What-if scenarios, stress testing, and global feature importance for deeper model understanding.")

    st.markdown("")

    # Probability explanation
    st.markdown("""
    <div class="info-box">
        <strong>📌 How to read predictions:</strong> The model outputs a probability between 0% and 100%.
        Higher <em>approval probability</em> (class 1 = "good credit") means the model estimates
        lower default risk. The default 50% threshold determines the binary approved/not-approved decision.
        These are model estimates, not guarantees.
    </div>
    """, unsafe_allow_html=True)

    # Recent results
    if st.session_state.get('prediction_result') is not None:
        st.markdown("---")
        st.markdown("##### 📋 Latest Analysis Result")
        result = st.session_state.prediction_result
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Model", result['model_name'])
        res_col2.metric("Approval Probability", f"{result['approval_probability']:.1%}")
        res_col3.metric("Risk Class", result['risk_class'])
        st.caption("Navigate to **Credit Risk Prediction** for details or **Reports & Downloads** for PDF export.")

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Educational Prototype</strong> — This application uses a historical German credit dataset
        (1994, 1,000 records) and should not be used for real lending decisions.
        See the "Fairness & Responsible AI" page for details.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: Credit Risk Prediction
# ==============================================================================
elif page == "👤 Credit Risk Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>Credit Risk Prediction</h1>
        <p class="subtitle">Enter applicant information for credit risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'applicant_data' not in st.session_state:
        st.session_state.applicant_data = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'shap_done' not in st.session_state:
        st.session_state.shap_done = False
    if 'lime_done' not in st.session_state:
        st.session_state.lime_done = False

    # Demo profile buttons
    st.markdown("##### 🎭 Quick Start — Demo Profiles")
    demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
    with demo_col1:
        if st.button("✅ Low-Risk Profile", use_container_width=True):
            st.session_state.demo_profile = 'low'
            st.session_state.prediction_result = None
            st.session_state.shap_done = False
            st.session_state.lime_done = False
            st.rerun()
    with demo_col2:
        if st.button("🟡 Moderate-Risk Profile", use_container_width=True):
            st.session_state.demo_profile = 'moderate'
            st.session_state.prediction_result = None
            st.session_state.shap_done = False
            st.session_state.lime_done = False
            st.rerun()
    with demo_col3:
        if st.button("🔴 High-Risk Profile", use_container_width=True):
            st.session_state.demo_profile = 'high'
            st.session_state.prediction_result = None
            st.session_state.shap_done = False
            st.session_state.lime_done = False
            st.rerun()
    with demo_col4:
        if st.button("🔄 Reset Form", use_container_width=True):
            for key in ['demo_profile', 'applicant_data', 'prediction_result', 'shap_done', 'lime_done',
                        'shap_values', 'shap_base_value', 'shap_feature_names', 'lime_exp']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Determine current profile values
    profile_key = st.session_state.get('demo_profile', 'low')
    if profile_key == 'high':
        defaults = HIGH_RISK_PROFILE
    elif profile_key == 'moderate':
        defaults = MODERATE_RISK_PROFILE
    else:
        defaults = LOW_RISK_PROFILE

    st.markdown("---")

    # Model selector
    model_choice = st.selectbox(
        "🤖 Select Model",
        ["XGBoost", "MLP"],
        help="XGBoost: Gradient boosting (primary model). MLP: Neural network (comparison)."
    )
    model_name = 'xgb_model' if model_choice == "XGBoost" else 'mlp_model'

    st.markdown("")

    # Input form
    applicant = get_applicant_form_inputs(defaults, key_prefix="pred_")

    st.markdown("---")

    # Analyze button
    if st.button("🔍 Analyze Applicant", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            try:
                result = make_prediction(applicant, model_name)
                if result:
                    st.session_state.prediction_result = result
                    st.session_state.applicant_data = applicant
                    st.session_state.model_name = model_name
                    st.session_state.shap_done = False
                    st.session_state.lime_done = False
                    # Clear old explainability results
                    for key in ['shap_values', 'shap_base_value', 'shap_feature_names', 'lime_exp']:
                        if key in st.session_state:
                            del st.session_state[key]
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Show prediction result
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        # Risk class badge
        risk = result['risk_class']
        if 'Low' in risk:
            badge_class = 'risk-badge-low'
            emoji = '✅'
        elif 'High' in risk:
            badge_class = 'risk-badge-high'
            emoji = '🔴'
        else:
            badge_class = 'risk-badge-medium'
            emoji = '🟡'

        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="risk-badge {badge_class}">{emoji} ESTIMATED {risk.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Visual probability bars
        render_probability_bar(result['approval_probability'], "Approval Probability")
        render_probability_bar(result['default_risk'], "Default Risk",
                             "linear-gradient(90deg, #EF5350, #E53935)" if result['default_risk'] > 0.5
                             else "linear-gradient(90deg, #FFA726, #FB8C00)" if result['default_risk'] > 0.3
                             else "linear-gradient(90deg, #66BB6A, #43A047)")

        st.markdown("")

        # Metrics
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Approval Probability", f"{result['approval_probability']:.1%}")
        rcol2.metric("Default Risk", f"{result['default_risk']:.1%}")
        rcol3.metric("Model", result['model_name'])
        rcol4.metric("Decision", "✅ Approved" if result['approved'] else "❌ Below Threshold")

        st.markdown("""
        <div class="info-box">
            <strong>📌 Note:</strong> The approval probability is the model's estimated
            likelihood of good credit for this specific applicant profile. Default risk = 1 − approval probability.
            The 50% threshold is a fixed project default, not an industry standard.
        </div>
        """, unsafe_allow_html=True)

        # Quick SHAP explanation (XGBoost only)
        if model_name == 'xgb_model':
            st.markdown("##### 💡 Why did the model make this prediction?")
            try:
                from src.explainability import get_shap_explainer, compute_shap_values, get_top_shap_features

                models = load_models()
                _, feature_names = load_preprocessor_cached()
                bg_samples = load_background_samples()

                xgb = models.get('xgb_model')
                if xgb:
                    explainer = get_shap_explainer(xgb, bg_samples)
                    shap_vals, base_val = compute_shap_values(explainer, result['X_processed'])
                    top_feats = get_top_shap_features(shap_vals, feature_names, top_n=5)

                    st.session_state.shap_values = shap_vals
                    st.session_state.shap_base_value = base_val
                    st.session_state.shap_feature_names = feature_names
                    st.session_state.shap_done = True

                    for feat in top_feats:
                        if 'approval' in feat['direction'].lower():
                            st.markdown(f"- 🟢 **{feat['feature']}** → {feat['direction']} (SHAP: {feat['value']:+.4f})")
                        else:
                            st.markdown(f"- 🔴 **{feat['feature']}** → {feat['direction']} (SHAP: {feat['value']:+.4f})")

                    st.caption("Navigate to **SHAP Explainability** for detailed waterfall and bar plots.")
            except Exception as e:
                st.warning(f"Could not generate quick explanation: {e}")
        else:
            st.info("💡 Quick SHAP explanations are available for XGBoost. Visit **LIME Explainability** for MLP explanations.")

        # PDF download right here
        st.markdown("---")
        st.markdown("##### 📄 Quick Report")
        if st.button("📥 Generate & Download PDF Report", use_container_width=True, key="pred_pdf"):
            with st.spinner("Generating PDF..."):
                try:
                    from reports.pdf_report import generate_pdf_report
                    display_data = format_applicant_for_display(st.session_state.applicant_data)
                    top_features = None
                    if st.session_state.get('shap_values') is not None:
                        from src.explainability import get_top_shap_features
                        top_features = get_top_shap_features(
                            st.session_state.shap_values,
                            st.session_state.shap_feature_names,
                            top_n=7
                        )
                    metrics = None
                    metrics_data = load_metrics_cached()
                    if metrics_data:
                        key = 'xgb' if 'xgb' in model_name else 'mlp'
                        metrics = metrics_data.get(key, {}).get('metrics', None)

                    pdf_bytes = generate_pdf_report(
                        applicant_data=display_data,
                        prediction_results=result,
                        top_features=top_features,
                        metrics=metrics,
                        model_name=result.get('model_name', 'XGBoost'),
                    )
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name=f"CreditLens_Report_{result.get('model_name', 'XGB')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        # Technical details
        with st.expander("🔧 Technical Details — Input Values"):
            display_data = format_applicant_for_display(st.session_state.applicant_data)
            st.dataframe(pd.DataFrame(list(display_data.items()), columns=['Feature', 'Value']),
                        use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Disclaimer:</strong> This prediction is an educational estimate based on a historical
            dataset. It does not represent a real lending decision or guarantee any outcome.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: SHAP Explainability
# ==============================================================================
elif page == "🌊 SHAP Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>SHAP Explainability</h1>
        <p class="subtitle">Understanding feature contributions with SHapley Additive exPlanations</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('prediction_result') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    result = st.session_state.prediction_result
    applicant = st.session_state.applicant_data
    model_name = st.session_state.get('model_name', 'xgb_model')

    st.markdown("""
    <div class="info-box">
        <strong>SHAP (SHapley Additive exPlanations)</strong> shows how individual features
        contribute to moving this prediction away from the model's baseline.
        Green bars push toward approval; red bars push toward default risk.
    </div>
    """, unsafe_allow_html=True)

    if model_name != 'xgb_model':
        st.warning("SHAP TreeExplainer is currently configured for XGBoost. Switch to XGBoost on the Credit Risk Prediction page for SHAP explanations.")
        st.stop()

    try:
        from src.explainability import (
            get_shap_explainer, compute_shap_values,
            create_shap_bar_plot, create_shap_waterfall_plot, get_top_shap_features
        )

        models = load_models()
        _, feature_names = load_preprocessor_cached()
        bg_samples = load_background_samples()

        xgb = models.get('xgb_model')
        if xgb is None:
            st.error("XGBoost model not loaded.")
            st.stop()

        # Compute or reuse SHAP values
        if st.session_state.get('shap_values') is not None:
            shap_vals = st.session_state.shap_values
            base_val = st.session_state.shap_base_value
        else:
            with st.spinner("Computing SHAP values..."):
                explainer = get_shap_explainer(xgb, bg_samples)
                shap_vals, base_val = compute_shap_values(explainer, result['X_processed'])
                st.session_state.shap_values = shap_vals
                st.session_state.shap_base_value = base_val
                st.session_state.shap_feature_names = feature_names

        st.session_state.shap_done = True

        # Context
        st.markdown(f"**Current prediction:** {result['approval_probability']:.1%} approval probability "
                   f"({result['risk_class']}) using {result['model_name']}")

        # Bar plot
        st.markdown("##### 📊 Feature Contributions")
        fig = create_shap_bar_plot(shap_vals, feature_names, top_n=12)
        st.pyplot(fig)
        plt.close(fig)

        # Waterfall plot
        st.markdown("##### 🌊 Waterfall Plot")
        try:
            fig2 = create_shap_waterfall_plot(
                shap_vals, base_val, feature_names,
                instance_values=result['X_processed'].flatten(),
                max_display=12
            )
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.warning(f"Waterfall plot not available: {e}")

        # Top features table
        st.markdown("##### 📋 Top Contributing Features")
        top_feats = get_top_shap_features(shap_vals, feature_names, top_n=12)
        feat_df = pd.DataFrame([
            {'Feature': f['feature'], 'SHAP Value': f"{f['value']:+.4f}",
             'Direction': f['direction'], '|Impact|': f"{f['abs_value']:.4f}"}
            for f in top_feats
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Human-readable summary
        st.markdown("##### 💡 Summary")
        pos_feats = [f for f in top_feats[:5] if f['value'] > 0]
        neg_feats = [f for f in top_feats[:5] if f['value'] < 0]
        if pos_feats:
            st.markdown("**Factors supporting approval:** " +
                       ", ".join([f"_{f['feature']}_" for f in pos_feats]))
        if neg_feats:
            st.markdown("**Factors increasing default risk:** " +
                       ", ".join([f"_{f['feature']}_" for f in neg_feats]))

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")


# ==============================================================================
# PAGE: LIME Explainability
# ==============================================================================
elif page == "🍋 LIME Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>LIME Explainability</h1>
        <p class="subtitle">Local Interpretable Model-agnostic Explanations</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('prediction_result') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    result = st.session_state.prediction_result
    model_name = st.session_state.get('model_name', 'xgb_model')

    st.markdown("""
    <div class="info-box">
        <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> creates a
        local linear approximation around this specific prediction using 1,200 perturbation samples.
        It works with any model (XGBoost or MLP).
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Current prediction:** {result['approval_probability']:.1%} approval probability "
               f"({result['risk_class']}) using {result['model_name']}")

    # Auto-generate or show existing
    generate = st.button("🍋 Generate LIME Explanation", type="primary", use_container_width=True)

    if generate or st.session_state.get('lime_done'):
        try:
            from src.explainability import explain_lime, get_lime_features, create_lime_plot

            models = load_models()
            preprocessor, feature_names = load_preprocessor_cached()
            training_samples = load_training_samples()

            model = models.get(model_name)
            if model is None:
                st.error(f"Model '{model_name}' not loaded.")
                st.stop()

            if generate or st.session_state.get('lime_exp') is None:
                with st.spinner("Generating LIME explanation (1,200 perturbations)..."):
                    lime_exp = explain_lime(
                        model, training_samples, feature_names,
                        result['X_processed'].flatten()
                    )
                    st.session_state.lime_exp = lime_exp
                    st.session_state.lime_done = True
            else:
                lime_exp = st.session_state.lime_exp

            # Plot
            st.markdown("##### 📊 Feature Contributions (LIME)")
            fig = create_lime_plot(lime_exp)
            st.pyplot(fig)
            plt.close(fig)

            # Table
            st.markdown("##### 📋 Feature Details")
            lime_feats = get_lime_features(lime_exp, top_n=10)
            lime_df = pd.DataFrame([
                {'Feature': f['feature'], 'Weight': f"{f['value']:+.4f}",
                 'Direction': f['direction'], '|Impact|': f"{f['abs_value']:.4f}"}
                for f in lime_feats
            ])
            st.dataframe(lime_df, use_container_width=True, hide_index=True)

            # Summary
            st.markdown("##### 💡 Summary")
            pos_feats = [f for f in lime_feats[:5] if f['value'] > 0]
            neg_feats = [f for f in lime_feats[:5] if f['value'] < 0]
            if pos_feats:
                st.markdown("**Factors supporting approval:** " +
                           ", ".join([f"_{f['feature']}_" for f in pos_feats]))
            if neg_feats:
                st.markdown("**Factors increasing default risk:** " +
                           ", ".join([f"_{f['feature']}_" for f in neg_feats]))

            st.markdown("""
            <div class="info-box">
                LIME builds a simple linear model around the prediction point. Larger positive weights
                push toward approval; negative weights push toward default risk. The linear approximation
                may differ from SHAP because it uses a different methodology.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"LIME explanation failed: {e}")


# ==============================================================================
# PAGE: SHAP vs LIME Comparison
# ==============================================================================
elif page == "⚖️ SHAP vs LIME":
    st.markdown("""
    <div class="main-header">
        <h1>SHAP vs LIME Comparison</h1>
        <p class="subtitle">Side-by-side explainability comparison</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('prediction_result') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    result = st.session_state.prediction_result
    model_name = st.session_state.get('model_name', 'xgb_model')

    st.markdown("""
    <div class="info-box">
        <strong>Descriptive comparison</strong> of SHAP and LIME explanations.
        This shows which top features are shared between both methods and whether
        they agree on the direction of influence.
    </div>
    """, unsafe_allow_html=True)

    if model_name != 'xgb_model':
        st.warning("SHAP vs LIME comparison requires XGBoost (SHAP TreeExplainer). Switch to XGBoost on the prediction page.")
        st.stop()

    st.markdown(f"**Prediction:** {result['approval_probability']:.1%} approval ({result['risk_class']}) — {result['model_name']}")

    # Auto-compute both if needed
    if st.button("⚖️ Run Comparison", type="primary", use_container_width=True):
        try:
            from src.explainability import (
                get_shap_explainer, compute_shap_values, get_top_shap_features,
                explain_lime, get_lime_features, compare_shap_lime,
                create_shap_bar_plot, create_lime_plot
            )

            models = load_models()
            _, feature_names = load_preprocessor_cached()
            bg_samples = load_background_samples()
            training_samples = load_training_samples()
            xgb = models.get('xgb_model')

            with st.spinner("Computing SHAP values..."):
                explainer = get_shap_explainer(xgb, bg_samples)
                shap_vals, base_val = compute_shap_values(explainer, result['X_processed'])
                st.session_state.shap_values = shap_vals
                st.session_state.shap_base_value = base_val
                st.session_state.shap_feature_names = feature_names
                st.session_state.shap_done = True

            with st.spinner("Computing LIME explanation (1,200 perturbations)..."):
                lime_exp = explain_lime(
                    xgb, training_samples, feature_names,
                    result['X_processed'].flatten()
                )
                st.session_state.lime_exp = lime_exp
                st.session_state.lime_done = True

            # Side-by-side plots
            st.markdown("##### 📊 Side-by-Side Contributions")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.markdown("**SHAP**")
                fig = create_shap_bar_plot(shap_vals, feature_names, top_n=8)
                st.pyplot(fig)
                plt.close(fig)
            with plot_col2:
                st.markdown("**LIME**")
                fig = create_lime_plot(lime_exp)
                st.pyplot(fig)
                plt.close(fig)

            # Comparison table
            st.markdown("##### 📋 Feature Comparison")
            shap_feats = get_top_shap_features(shap_vals, feature_names, top_n=7)
            lime_feats = get_lime_features(lime_exp, top_n=7)
            comparison = compare_shap_lime(shap_feats, lime_feats, top_n=7)

            st.markdown(f"**Shared top features:** {comparison['shared_features']} / {comparison['total_compared']}")

            if comparison.get('shap_only'):
                st.markdown(f"**SHAP-only features:** {', '.join(comparison['shap_only'])}")
            if comparison.get('lime_only'):
                st.markdown(f"**LIME-only features:** {', '.join(comparison['lime_only'])}")

            comp_df = pd.DataFrame(comparison['rows'])
            comp_df.columns = ['Feature', 'SHAP Direction', 'LIME Direction', 'Directions Agree']
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Note:</strong> This is a descriptive comparison — not an accuracy or reliability score.
                SHAP uses game-theoretic attribution while LIME uses local linear approximation.
                They may highlight different aspects because they measure feature importance differently.
                Disagreements are expected and do not indicate an error.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Comparison failed: {e}")

    elif st.session_state.get('shap_done') and st.session_state.get('lime_done'):
        st.info("Previous comparison results available. Click **Run Comparison** to recompute.")


# ==============================================================================
# PAGE: What-if Analysis
# ==============================================================================
elif page == "🧪 What-if Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>What-if Scenario Analysis</h1>
        <p class="subtitle">How does changing inputs affect the model's prediction?</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('applicant_data') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important:</strong> This is an illustrative bounded sensitivity analysis.
        It is NOT financial advice, a guaranteed approval path, an optimal loan recommendation,
        or a formal counterfactual explanation.
    </div>
    """, unsafe_allow_html=True)

    applicant = st.session_state.applicant_data
    current_result = st.session_state.prediction_result
    model_name = st.session_state.get('model_name', 'xgb_model')

    st.markdown(f"**Current prediction:** {current_result['approval_probability']:.1%} approval probability "
               f"({current_result['risk_class']})")

    st.markdown("---")

    # Variable selectors
    st.markdown("##### 📝 Adjust Scenario Variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Credit Amount**")
        base_amt = int(applicant.get('credit_amount', 5000))
        new_amt = st.number_input("New Credit Amount (DM)", min_value=250, max_value=20000,
                                  value=base_amt, step=250, key="wi_amt")
        st.caption(f"Original: {base_amt} DM")

    with col2:
        st.markdown("**Loan Duration**")
        base_dur = int(applicant.get('duration', 24))
        new_dur = st.number_input("New Duration (months)", min_value=4, max_value=72,
                                  value=base_dur, step=1, key="wi_dur")
        st.caption(f"Original: {base_dur} months")

    with col3:
        st.markdown("**Age**")
        base_age = int(applicant.get('age', 35))
        new_age = st.number_input("New Age (years)", min_value=18, max_value=80,
                                  value=base_age, step=1, key="wi_age")
        st.caption(f"Original: {base_age} years")

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        if st.button("🔍 Run What-if Scenario", type="primary", use_container_width=True):
            with st.spinner("Computing scenario..."):
                try:
                    # Build modified applicant
                    modified = copy.deepcopy(applicant)
                    modified['credit_amount'] = new_amt
                    modified['duration'] = new_dur
                    modified['age'] = new_age

                    new_result = make_prediction(modified, model_name)
                    if new_result:
                        st.session_state.whatif_result = new_result
                        st.session_state.whatif_modified = modified
                except Exception as e:
                    st.error(f"What-if analysis failed: {e}")
    with wi_col2:
        if st.button("🔄 Reset Scenario", use_container_width=True):
            for key in ['whatif_result', 'whatif_modified']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Show results
    if st.session_state.get('whatif_result') is not None:
        wi_result = st.session_state.whatif_result
        st.markdown("---")
        st.markdown("##### 📊 Scenario Comparison")

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            st.metric("Original Approval", f"{current_result['approval_probability']:.1%}")
        with comp_col2:
            delta = wi_result['approval_probability'] - current_result['approval_probability']
            st.metric("Scenario Approval", f"{wi_result['approval_probability']:.1%}",
                     delta=f"{delta:+.1%}")
        with comp_col3:
            st.metric("Scenario Risk Class", wi_result['risk_class'])

        # Changes summary
        changes = []
        if new_amt != base_amt:
            changes.append(f"Credit Amount: {base_amt} → {new_amt} DM")
        if new_dur != base_dur:
            changes.append(f"Duration: {base_dur} → {new_dur} months")
        if new_age != base_age:
            changes.append(f"Age: {base_age} → {new_age} years")
        if changes:
            st.markdown("**Changes made:** " + " | ".join(changes))

        # Sensitivity charts
        st.markdown("##### 📈 Sensitivity Curves")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            amounts = np.linspace(max(250, base_amt - 5000), min(20000, base_amt + 5000), 15).astype(int)
            amt_probs = []
            for amt in amounts:
                app = copy.deepcopy(applicant)
                app['credit_amount'] = int(amt)
                r = make_prediction(app, model_name)
                amt_probs.append(r['approval_probability'] if r else 0)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(amounts, amt_probs, color='#4FC3F7', linewidth=2, marker='o', markersize=3)
            ax.axhline(y=THRESHOLD, color='#EF5350', linestyle='--', alpha=0.7, label=f'Threshold ({THRESHOLD:.0%})')
            ax.axvline(x=base_amt, color='#66BB6A', linestyle='--', alpha=0.7, label=f'Current ({base_amt})')
            ax.set_xlabel('Credit Amount (DM)')
            ax.set_ylabel('Approval Probability')
            ax.set_title('Impact of Credit Amount')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart_col2:
            durations = list(range(max(6, base_dur - 18), min(72, base_dur + 24), 3))
            dur_probs = []
            for dur in durations:
                app = copy.deepcopy(applicant)
                app['duration'] = int(dur)
                r = make_prediction(app, model_name)
                dur_probs.append(r['approval_probability'] if r else 0)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(durations, dur_probs, color='#FFA726', linewidth=2, marker='o', markersize=3)
            ax.axhline(y=THRESHOLD, color='#EF5350', linestyle='--', alpha=0.7, label=f'Threshold ({THRESHOLD:.0%})')
            ax.axvline(x=base_dur, color='#66BB6A', linestyle='--', alpha=0.7, label=f'Current ({base_dur})')
            ax.set_xlabel('Duration (months)')
            ax.set_ylabel('Approval Probability')
            ax.set_title('Impact of Loan Duration')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ==============================================================================
# PAGE: Stress Testing
# ==============================================================================
elif page == "⚠️ Stress Testing":
    st.markdown("""
    <div class="main-header">
        <h1>Stress Testing</h1>
        <p class="subtitle">One-variable-at-a-time sensitivity analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('applicant_data') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    st.markdown("""
    <div class="info-box">
        This shows how the model's prediction changes when a single variable varies
        across its range while all other variables remain at their current values.
        This is a one-variable-at-a-time sensitivity analysis — not a Partial Dependence Plot.
    </div>
    """, unsafe_allow_html=True)

    applicant = st.session_state.applicant_data
    model_name = st.session_state.get('model_name', 'xgb_model')

    stress_col1, stress_col2 = st.columns([2, 1])
    with stress_col1:
        variable = st.selectbox("Select Variable to Test",
                               ['credit_amount', 'duration', 'age', 'installment_rate', 'existing_credits'],
                               format_func=lambda x: get_feature_display_name(x))
    with stress_col2:
        compare_models = st.checkbox("Compare both models", value=False,
                                     help="Run stress test on both XGBoost and MLP")

    if st.button("▶️ Run Stress Test", type="primary", use_container_width=True):
        with st.spinner("Running stress test..."):
            try:
                from src.stress_testing import DEFAULT_RANGES

                extended_ranges = {
                    'credit_amount': np.linspace(250, 18500, 25).astype(int).tolist(),
                    'duration': list(range(6, 73, 3)),
                    'age': list(range(19, 76, 2)),
                    'installment_rate': [1, 2, 3, 4],
                    'existing_credits': [1, 2, 3, 4],
                }

                values = extended_ranges.get(variable, DEFAULT_RANGES.get(variable, list(range(1, 50))))

                models_to_test = ['xgb_model', 'mlp_model'] if compare_models else [model_name]
                all_results = {}

                for mn in models_to_test:
                    results = []
                    for val in values:
                        app = copy.deepcopy(applicant)
                        app[variable] = val
                        r = make_prediction(app, mn)
                        if r:
                            results.append({variable: val, 'approval_probability': r['approval_probability']})
                    all_results[mn] = pd.DataFrame(results)

                # Chart
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = {'xgb_model': '#4FC3F7', 'mlp_model': '#FFA726'}
                labels = {'xgb_model': 'XGBoost', 'mlp_model': 'MLP'}

                for mn, df in all_results.items():
                    ax.plot(df[variable], df['approval_probability'],
                           color=colors.get(mn, '#4FC3F7'), linewidth=2.5,
                           marker='o', markersize=3, label=labels.get(mn, mn))

                ax.axhline(y=THRESHOLD, color='#EF5350', linestyle='--', linewidth=1.5,
                          alpha=0.8, label=f'Threshold ({THRESHOLD:.0%})')

                current_val = applicant.get(variable)
                if current_val is not None:
                    ax.axvline(x=float(current_val), color='#66BB6A', linestyle='--',
                              linewidth=1.5, alpha=0.8, label=f'Current ({current_val})')

                primary_df = all_results.get(model_name, list(all_results.values())[0])
                ax.fill_between(primary_df[variable], primary_df['approval_probability'],
                               THRESHOLD, alpha=0.08, color='#4FC3F7')

                ax.set_xlabel(get_feature_display_name(variable), fontsize=11)
                ax.set_ylabel('Approval Probability', fontsize=11)
                ax.set_title(f'Sensitivity: {get_feature_display_name(variable)}',
                           fontsize=13, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Stats
                st.markdown("##### 📋 Summary Statistics")
                for mn, df in all_results.items():
                    label = labels.get(mn, mn)
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    scol1.metric(f"{label} — Min", f"{df['approval_probability'].min():.1%}")
                    scol2.metric(f"{label} — Max", f"{df['approval_probability'].max():.1%}")
                    scol3.metric(f"{label} — Range",
                               f"{(df['approval_probability'].max() - df['approval_probability'].min()):.1%}")
                    scol4.metric(f"{label} — Std Dev", f"{df['approval_probability'].std():.3f}")

                st.markdown("""
                <div class="info-box">
                    <strong>Interpretation:</strong> A wider range or higher standard deviation indicates
                    the model is more sensitive to this variable. Other variables are held constant,
                    so this analysis does not capture interaction effects.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Stress test failed: {e}")


# ==============================================================================
# PAGE: Global Feature Importance
# ==============================================================================
elif page == "🌍 Global Feature Importance":
    st.markdown("""
    <div class="main-header">
        <h1>Global Feature Importance</h1>
        <p class="subtitle">Which features influence the model most across all predictions?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Global SHAP feature importance shows which features have the largest average impact
        on model predictions across many samples. This shows model association/importance,
        <strong>not causation</strong>.
    </div>
    """, unsafe_allow_html=True)

    if st.button("📊 Compute Global Feature Importance", type="primary", use_container_width=True):
        with st.spinner("Computing global SHAP importance (this may take a moment)..."):
            try:
                from src.explainability import create_global_shap_importance, _clean_feature_name

                models = load_models()
                _, feature_names = load_preprocessor_cached()
                bg_samples = load_background_samples()

                xgb = models.get('xgb_model')
                if xgb is None:
                    st.error("XGBoost model not loaded.")
                    st.stop()

                fig, importance_list = create_global_shap_importance(
                    xgb, bg_samples, feature_names, top_n=15
                )
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("##### 📋 Top Features by Mean |SHAP Value|")
                imp_df = pd.DataFrame(importance_list, columns=['Feature', 'Mean |SHAP|'])
                imp_df['Feature'] = imp_df['Feature'].apply(_clean_feature_name)
                imp_df['Mean |SHAP|'] = imp_df['Mean |SHAP|'].apply(lambda x: f"{x:.4f}")
                imp_df['Rank'] = range(1, len(imp_df) + 1)
                imp_df = imp_df[['Rank', 'Feature', 'Mean |SHAP|']]
                st.dataframe(imp_df, use_container_width=True, hide_index=True)

                # XGBoost built-in importance comparison
                st.markdown("##### 🌲 XGBoost Built-in Feature Importance")
                try:
                    xgb_importance = xgb.feature_importances_
                    top_idx = np.argsort(xgb_importance)[-15:]
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(range(len(top_idx)),
                            [xgb_importance[i] for i in top_idx],
                            color='#00BCD4', edgecolor='none', height=0.6)
                    labels = [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in top_idx]
                    ax2.set_yticks(range(len(top_idx)))
                    ax2.set_yticklabels([_clean_feature_name(l) for l in labels], fontsize=9)
                    ax2.set_xlabel('Feature Importance (Gain)', fontsize=10)
                    ax2.set_title('XGBoost Built-in Feature Importance', fontsize=12, fontweight='bold')
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Global insights failed: {e}")

    st.markdown("""
    <div class="warning-box">
        Feature importance reflects the model's learned patterns from the training data.
        It does not prove causal relationships or imply that changing a feature
        will directly change someone's credit worthiness.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: Model Performance
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>Model Performance & Benchmarks</h1>
        <p class="subtitle">Comparing XGBoost and MLP on the held-out test set</p>
    </div>
    """, unsafe_allow_html=True)

    metrics_data = load_metrics_cached()
    manifest = load_manifest_cached()

    if not metrics_data:
        st.error("No metrics found. Run train.py first.")
        st.stop()

    xgb_m = metrics_data.get('xgb', {}).get('metrics', {})
    mlp_m = metrics_data.get('mlp', {}).get('metrics', {})

    # Dataset info
    st.markdown("""
    <div class="info-box">
        Both models are evaluated on the <strong>same 200-record held-out test set</strong>
        (80/20 stratified split, random_state=42). Metrics are computed from actual model predictions
        on data not seen during training.
    </div>
    """, unsafe_allow_html=True)

    # Metrics comparison table
    st.markdown("##### 📊 Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'XGBoost': [
            f"{xgb_m.get('accuracy', 0)*100:.1f}%",
            f"{xgb_m.get('precision', 0)*100:.1f}%",
            f"{xgb_m.get('recall', 0)*100:.1f}%",
            f"{xgb_m.get('f1', 0):.3f}",
            f"{xgb_m.get('roc_auc', 0):.3f}",
        ],
        'MLP': [
            f"{mlp_m.get('accuracy', 0)*100:.1f}%",
            f"{mlp_m.get('precision', 0)*100:.1f}%",
            f"{mlp_m.get('recall', 0)*100:.1f}%",
            f"{mlp_m.get('f1', 0):.3f}",
            f"{mlp_m.get('roc_auc', 0):.3f}",
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion matrices and ROC curves
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrices", "📈 ROC Curves", "📉 Precision-Recall"])

    with tab1:
        cm_col1, cm_col2 = st.columns(2)

        for col, name, data_key in [(cm_col1, 'XGBoost', 'xgb'), (cm_col2, 'MLP', 'mlp')]:
            with col:
                cm = metrics_data.get(data_key, {}).get('confusion_matrix', [[0,0],[0,0]])
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

                for i in range(2):
                    for j in range(2):
                        color = 'white' if cm[i][j] > max(max(cm))/2 else 'black'
                        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                               color=color, fontsize=14, fontweight='bold')

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Default (0)', 'Good (1)'])
                ax.set_yticklabels(['Default (0)', 'Good (1)'])
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                ax.set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax, shrink=0.8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 6))

        for data_key, name, color in [('xgb', 'XGBoost', '#4FC3F7'), ('mlp', 'MLP', '#FFA726')]:
            roc = metrics_data.get(data_key, {}).get('roc', {})
            fpr = roc.get('fpr', [0, 1])
            tpr = roc.get('tpr', [0, 1])
            auc_val = metrics_data.get(data_key, {}).get('metrics', {}).get('roc_auc', 0)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves — XGBoost vs MLP', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="info-box">
            <strong>ROC Curve:</strong> Shows the trade-off between true-positive rate and
            false-positive rate at different classification thresholds.<br>
            <strong>AUC:</strong> Summarizes the area under the ROC curve (1.0 = perfect, 0.5 = random).
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="info-box">
            Precision-recall analysis uses saved confusion matrix data. For the positive class
            (class 1 = good credit), precision measures the proportion of predicted positives
            that are truly positive, while recall measures how many actual positives are captured.
        </div>
        """, unsafe_allow_html=True)

        # Model comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        x = np.arange(len(labels))
        width = 0.35

        xgb_vals = [xgb_m.get(m, 0) for m in metrics_list]
        mlp_vals = [mlp_m.get(m, 0) for m in metrics_list]

        bars1 = ax.bar(x - width/2, xgb_vals, width, label='XGBoost', color='#4FC3F7', edgecolor='none')
        bars2 = ax.bar(x + width/2, mlp_vals, width, label='MLP', color='#FFA726', edgecolor='none')

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Model Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='y')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Methodology
    st.markdown("---")
    st.markdown("##### 📋 Evaluation Methodology")
    method_col1, method_col2 = st.columns(2)
    with method_col1:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | **Split** | 80% train / 20% test |
        | **Train samples** | {manifest.get('train_size', 800) if manifest else 800} |
        | **Test samples** | {manifest.get('test_size', 200) if manifest else 200} |
        | **Stratification** | Yes (preserves class ratio) |
        | **Random state** | {RANDOM_STATE} |
        """)
    with method_col2:
        st.markdown("""
        | Detail | Description |
        |--------|-------------|
        | **Positive class** | Class 1 (good credit / approval) |
        | **Negative class** | Class 0 (bad credit / default) |
        | **Threshold** | 50% (fixed, not optimized) |
        | **Cross-validation** | Not used (single split) |
        | **Calibration** | Not applied |
        """)


# ==============================================================================
# PAGE: Fairness & Responsible AI
# ==============================================================================
elif page == "🛡️ Fairness & Responsible AI":
    st.markdown("""
    <div class="main-header">
        <h1>Fairness, Limitations & Responsible AI</h1>
        <p class="subtitle">Understanding the scope and limitations of this prototype</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["⚠️ Limitations", "🛡️ Fairness", "🔒 Privacy", "📜 Responsible Use"])

    with tab1:
        st.markdown("##### Known Limitations")
        st.markdown("""
        **Dataset Limitations:**
        - Only 1,000 records — small by modern standards
        - Historical German credit data from **1994**
        - Monetary values in Deutsche Mark (DM), not current currency
        - May not reflect current credit patterns or market conditions
        - No representation of modern financial products or behaviors

        **Methodology Limitations:**
        - Single 80/20 train-test split — no k-fold cross-validation
        - No automated hyperparameter tuning (no Optuna, no grid search)
        - No model calibration (probabilities may not be well-calibrated)
        - Fixed 50% classification threshold (not optimized for business objectives)
        - No formal fairness evaluation or bias testing
        - No feature selection or dimensionality reduction applied

        **Explainability Limitations:**
        - SHAP TreeExplainer is currently used only for XGBoost
        - LIME is an approximate local explanation that may vary between runs
        - What-if analysis is bounded illustrative sensitivity analysis
        - Stress testing is one-variable-at-a-time (interactions not captured)
        - Feature importance does **not** establish causation
        """)

    with tab2:
        st.markdown("##### Fairness Considerations")
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ This model has NOT been evaluated for fairness or bias.</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        - The historical dataset contains sensitive attributes (e.g., personal status/sex, foreign worker status)
        - The model may have learned patterns that reflect **historical biases** in lending
        - Formal fairness metrics (equalized odds, demographic parity, calibration across groups) have **not** been implemented
        - No disparate impact analysis has been performed
        - This project should **not** be used for real lending decisions

        **If this were a production system**, it would require:
        - Formal bias testing across protected groups
        - Compliance with fair lending regulations (ECOA, FCRA, etc.)
        - Regular monitoring and auditing
        - Human oversight in all decisions
        - Model validation by independent reviewers
        - Adverse action notice generation
        """)

    with tab3:
        st.markdown("##### Privacy & Data Handling")
        st.markdown("""
        - This prototype uses **fictional demonstration profiles** by default
        - No user data is collected, stored, or transmitted to external services
        - All processing happens locally in the browser session
        - Session data is temporary and cleared on page refresh
        - The training dataset is publicly available from UCI/OpenML
        - No personally identifiable information (PII) is used

        **Best practices if extending this project:**
        - Never log applicant input data
        - Never send applicant data to third-party APIs without consent
        - Implement data retention policies
        - Follow applicable data protection regulations
        """)

    with tab4:
        st.markdown("##### Responsible Use Guidelines")
        st.markdown("""
        <div class="success-box">
            <strong>✅ Appropriate uses:</strong>
            <ul>
                <li>Educational demonstration of ML credit risk concepts</li>
                <li>Learning about model explainability (SHAP, LIME)</li>
                <li>Portfolio project showcasing full-stack ML skills</li>
                <li>Understanding sensitivity analysis and stress testing</li>
                <li>Academic research and coursework</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            <strong>❌ NOT appropriate for:</strong>
            <ul>
                <li>Approving or denying real credit applications</li>
                <li>Making any real financial decisions</li>
                <li>Claiming regulatory compliance or certification</li>
                <li>Production deployment without extensive validation</li>
                <li>Inferring real-world creditworthiness of any individual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: Reports & Downloads
# ==============================================================================
elif page == "📄 Reports & Downloads":
    st.markdown("""
    <div class="main-header">
        <h1>Reports & Downloads</h1>
        <p class="subtitle">Generate and download analysis reports</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('prediction_result') is None:
        st.info("👈 Please analyze an applicant first on the **Credit Risk Prediction** page.")
        st.stop()

    result = st.session_state.prediction_result
    applicant = st.session_state.applicant_data
    model_name = st.session_state.get('model_name', 'xgb_model')

    # Preview
    st.markdown("##### 📋 Report Preview")

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Approval Probability", f"{result['approval_probability']:.1%}")
    pcol2.metric("Risk Class", result['risk_class'])
    pcol3.metric("Model", result['model_name'])

    st.markdown("##### 📄 Report Contents")
    st.markdown("""
    - CreditLens AI header with generation date
    - Selected model and version information
    - Applicant information table (all input features)
    - Prediction results (approval probability, default risk, risk class)
    - Top feature contributions from SHAP (if available)
    - Model performance metrics (from held-out test set)
    - Educational disclaimer and responsible AI notice
    """)

    st.markdown("---")

    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF..."):
            try:
                from reports.pdf_report import generate_pdf_report

                display_data = format_applicant_for_display(applicant)

                # Get top features if available
                top_features = None
                if st.session_state.get('shap_values') is not None:
                    from src.explainability import get_top_shap_features
                    top_features = get_top_shap_features(
                        st.session_state.shap_values,
                        st.session_state.shap_feature_names,
                        top_n=7
                    )

                # Get metrics
                metrics = None
                metrics_data = load_metrics_cached()
                if metrics_data:
                    key = 'xgb' if 'xgb' in model_name else 'mlp'
                    metrics = metrics_data.get(key, {}).get('metrics', None)

                pdf_bytes = generate_pdf_report(
                    applicant_data=display_data,
                    prediction_results=result,
                    top_features=top_features,
                    metrics=metrics,
                    model_name=result.get('model_name', 'XGBoost'),
                )

                st.success("✅ PDF report generated successfully!")

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"CreditLens_Report_{result.get('model_name', 'XGB')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    # Data export
    st.markdown("---")
    st.markdown("##### 📊 Data Export")

    if st.button("📊 Export Applicant Data (CSV)", use_container_width=True):
        display_data = format_applicant_for_display(applicant)
        export_df = pd.DataFrame(list(display_data.items()), columns=['Feature', 'Value'])
        export_df.loc[len(export_df)] = ['Approval Probability', f"{result['approval_probability']:.4f}"]
        export_df.loc[len(export_df)] = ['Default Risk', f"{result['default_risk']:.4f}"]
        export_df.loc[len(export_df)] = ['Risk Class', result['risk_class']]
        export_df.loc[len(export_df)] = ['Model', result['model_name']]

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"CreditLens_Data_{result.get('model_name', 'XGB')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==============================================================================
# PAGE: About the Project
# ==============================================================================
elif page == "ℹ️ About the Project":
    st.markdown("""
    <div class="main-header">
        <h1>About CreditLens AI</h1>
        <p class="subtitle">Project information, architecture, and technology stack</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🏗️ Architecture", "📚 References"])

    with tab1:
        st.markdown("##### Project Summary")
        st.markdown("""
        **CreditLens AI** is an educational prototype for explainable credit risk analysis.
        It demonstrates how machine learning models can be used to predict credit risk
        while providing transparent explanations of their decisions.

        | Component | Details |
        |-----------|---------|
        | **Project** | CreditLens AI |
        | **Type** | Educational Explainable Credit Risk Prototype |
        | **Dataset** | UCI Statlog German Credit (1,000 records, 1994) |
        | **Models** | XGBoost (gradient boosting) + MLP (neural network) |
        | **Explainability** | SHAP (TreeExplainer) + LIME |
        | **Frontend** | Streamlit |
        | **Reporting** | fpdf2 (PDF generation) |
        | **Testing** | unittest |
        | **Language** | Python 3.9+ |
        """)

        st.markdown("##### Technology Stack")
        cols = st.columns(4)
        techs = [
            ("🐍 Python", "Core language"),
            ("📊 Streamlit", "Dashboard framework"),
            ("🌲 XGBoost", "Gradient boosting model"),
            ("🧠 scikit-learn", "MLP + preprocessing"),
            ("💡 SHAP", "Feature attribution"),
            ("🍋 LIME", "Local explanations"),
            ("📈 Matplotlib", "Data visualization"),
            ("📄 fpdf2", "PDF report generation"),
        ]
        for i, (name, desc) in enumerate(techs):
            with cols[i % 4]:
                st.markdown(f"**{name}**<br><span style='color:#90A4AE;font-size:0.8rem'>{desc}</span>",
                           unsafe_allow_html=True)

        st.markdown("##### Features")
        st.markdown("""
        1. **Credit Risk Prediction** — XGBoost and MLP predictions with probability estimates
        2. **SHAP Explainability** — Feature contribution analysis with waterfall and bar plots
        3. **LIME Explainability** — Model-agnostic local explanations
        4. **SHAP vs LIME Comparison** — Side-by-side method comparison
        5. **What-if Scenario Analysis** — Explore how changes affect predictions
        6. **Stress Testing** — Sensitivity analysis across feature ranges
        7. **Global Feature Importance** — Dataset-level feature analysis
        8. **Model Performance** — Detailed evaluation metrics and visualizations
        9. **PDF Reports** — Professional downloadable reports
        10. **Responsible AI** — Fairness, limitations, and privacy documentation
        """)

    with tab2:
        st.markdown("##### Project Architecture")
        st.markdown("""
        ```
        CreditLens AI/
        ├── app.py                    # Main Streamlit application
        ├── config.py                 # Central configuration
        ├── train.py                  # Model training script
        ├── requirements.txt          # Python dependencies
        ├── run.sh                    # Launcher script
        ├── .streamlit/config.toml    # Streamlit theme configuration
        ├── src/                      # Backend modules
        │   ├── data.py              # Dataset loading (UCI German Credit)
        │   ├── preprocessing.py     # Feature preprocessing pipeline
        │   ├── prediction.py        # Prediction engine
        │   ├── evaluation.py        # Model evaluation metrics
        │   ├── explainability.py    # SHAP + LIME explanations
        │   ├── feature_mapping.py   # UCI code → human-readable labels
        │   ├── what_if.py           # What-if scenario analysis
        │   └── stress_testing.py    # Sensitivity analysis
        ├── models/                   # Saved model artifacts
        │   ├── xgb_model.joblib     # Trained XGBoost model
        │   ├── mlp_model.joblib     # Trained MLP model
        │   ├── preprocessor.joblib  # Fitted preprocessing pipeline
        │   ├── feature_names.joblib # Transformed feature names
        │   ├── background_samples.joblib  # SHAP background data
        │   ├── test_samples.joblib  # LIME training data
        │   ├── metrics.json         # Evaluation metrics
        │   └── model_manifest.json  # Artifact metadata
        ├── reports/                  # Report generation
        │   └── pdf_report.py        # PDF report generator
        └── tests/                    # Test suite
            └── test_all.py          # Comprehensive tests
        ```
        """)

        st.markdown("##### Data Pipeline")
        st.markdown("""
        1. **Data Loading**: UCI German Credit dataset from OpenML (1,000 records, 20 features)
        2. **Target Mapping**: `good` → 1 (approval), `bad` → 0 (default)
        3. **Feature Types**: 7 numeric features scaled with StandardScaler, 13 categorical features one-hot encoded
        4. **Preprocessing**: sklearn ColumnTransformer produces 61 transformed features
        5. **Train/Test Split**: 80/20 stratified split (random_state=42)
        6. **Models**: XGBoost (150 trees, depth 4) and MLP (64→32, ReLU, Adam)
        7. **Artifacts**: All models and preprocessors saved to `models/` directory
        """)

    with tab3:
        st.markdown("##### Dataset Reference")
        st.markdown("""
        **UCI Statlog German Credit Dataset**
        - Source: UCI Machine Learning Repository (OpenML ID: 31)
        - Records: 1,000
        - Features: 20 (7 numeric, 13 categorical)
        - Year: 1994
        - Currency: Deutsche Mark (DM)
        - [OpenML Dataset Page](https://www.openml.org/d/31)
        - [UCI Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

        **Key Libraries**
        - [SHAP Documentation](https://shap.readthedocs.io/)
        - [LIME Documentation](https://lime-ml.readthedocs.io/)
        - [XGBoost Documentation](https://xgboost.readthedocs.io/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        """)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Educational Prototype</strong> — This application is designed for learning
            and demonstration purposes. It should not be used for real lending decisions.
        </div>
        """, unsafe_allow_html=True)
