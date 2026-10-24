"""
CreditLens AI - Feature Mapping Module
=======================================
Maps UCI German Credit dataset codes to human-readable descriptions.
All mappings based on the official UCI dataset documentation.
"""

# ==============================================================================
# UCI Code → Human-Readable Value Mappings
# ==============================================================================
FEATURE_VALUE_MAPPINGS = {
    'checking_account': {
        "'<0'": '< 0 DM (overdrawn)',
        "'0<=X<200'": '0 – 200 DM',
        "'>=200'": '≥ 200 DM',
        "'no checking'": 'No checking account',
        # Legacy A-codes (fallback)
        'A11': '< 0 DM (overdrawn)',
        'A12': '0 – 200 DM',
        'A13': '≥ 200 DM',
        'A14': 'No checking account',
    },
    'credit_history': {
        "'no credits/all paid'": 'No credits / all paid',
        "'all paid'": 'All paid duly',
        "'existing paid'": 'Existing credits paid duly',
        "'delayed previously'": 'Delayed payment in past',
        "'critical/other existing credit'": 'Critical account / other credits',
        # Legacy A-codes
        'A30': 'No credits taken / all paid',
        'A31': 'All credits paid back duly',
        'A32': 'Existing credits paid duly',
        'A33': 'Delay in paying off in past',
        'A34': 'Critical account / other credits',
    },
    'purpose': {
        "'new car'": 'New car',
        "'used car'": 'Used car',
        'furniture/equipment': 'Furniture / equipment',
        'radio/tv': 'Radio / television',
        "'domestic appliance'": 'Domestic appliance',
        'repairs': 'Repairs',
        'education': 'Education',
        'vacation': 'Vacation',
        'retraining': 'Retraining',
        'business': 'Business',
        'other': 'Other',
        # Legacy A-codes
        'A40': 'New car',
        'A41': 'Used car',
        'A42': 'Furniture / equipment',
        'A43': 'Radio / television',
        'A44': 'Domestic appliance',
        'A45': 'Repairs',
        'A46': 'Education',
        'A47': 'Vacation',
        'A48': 'Retraining',
        'A49': 'Business',
        'A410': 'Other',
    },
    'savings_account': {
        "'<100'": '< 100 DM',
        "'100<=X<500'": '100 – 500 DM',
        "'500<=X<1000'": '500 – 1000 DM',
        "'>=1000'": '≥ 1000 DM',
        "'no known savings'": 'No known savings',
        # Legacy A-codes
        'A61': '< 100 DM',
        'A62': '100 – 500 DM',
        'A63': '500 – 1000 DM',
        'A64': '≥ 1000 DM',
        'A65': 'No known savings',
    },
    'employment_since': {
        'unemployed': 'Unemployed',
        "'<1'": '< 1 year',
        "'1<=X<4'": '1 – 4 years',
        "'4<=X<7'": '4 – 7 years',
        "'>=7'": '≥ 7 years',
        # Legacy A-codes
        'A71': 'Unemployed',
        'A72': '< 1 year',
        'A73': '1 – 4 years',
        'A74': '4 – 7 years',
        'A75': '≥ 7 years',
    },
    'personal_status_sex': {
        "'male div/sep'": 'Male: divorced / separated',
        "'female div/dep/mar'": 'Female: divorced / separated / married',
        "'male single'": 'Male: single',
        "'male mar/wid'": 'Male: married / widowed',
        "'female single'": 'Female: single',
        # Legacy A-codes
        'A91': 'Male: divorced / separated',
        'A92': 'Female: divorced / separated / married',
        'A93': 'Male: single',
        'A94': 'Male: married / widowed',
        'A95': 'Female: single',
    },
    'other_debtors': {
        'none': 'None',
        "'co applicant'": 'Co-applicant',
        'guarantor': 'Guarantor',
        # Legacy A-codes
        'A101': 'None',
        'A102': 'Co-applicant',
        'A103': 'Guarantor',
    },
    'property': {
        "'real estate'": 'Real estate',
        "'life insurance'": 'Building society savings / life insurance',
        'car': 'Car or other',
        "'no known property'": 'No known property',
        # Legacy A-codes
        'A121': 'Real estate',
        'A122': 'Building society savings / life insurance',
        'A123': 'Car or other',
        'A124': 'No known property',
    },
    'other_installment_plans': {
        'bank': 'Bank',
        'stores': 'Stores',
        'none': 'None',
        # Legacy A-codes
        'A141': 'Bank',
        'A142': 'Stores',
        'A143': 'None',
    },
    'housing': {
        'rent': 'Rent',
        'own': 'Own',
        "'for free'": 'For free',
        # Legacy A-codes
        'A151': 'Rent',
        'A152': 'Own',
        'A153': 'For free',
    },
    'job': {
        "'unemp/unskilled non res'": 'Unemployed / unskilled non-resident',
        "'unskilled resident'": 'Unskilled resident',
        'skilled': 'Skilled employee / official',
        "'high qualif/self emp/mgmt'": 'Management / self-employed / highly qualified',
        # Legacy A-codes
        'A171': 'Unemployed / unskilled non-resident',
        'A172': 'Unskilled resident',
        'A173': 'Skilled employee / official',
        'A174': 'Management / self-employed / highly qualified',
    },
    'telephone': {
        'none': 'None',
        'yes': 'Yes (registered)',
        # Legacy A-codes
        'A191': 'None',
        'A192': 'Yes (registered)',
    },
    'foreign_worker': {
        'yes': 'Yes',
        'no': 'No',
        # Legacy A-codes
        'A201': 'Yes',
        'A202': 'No',
    },
}

# ==============================================================================
# Human-Readable Feature Names
# ==============================================================================
FEATURE_DISPLAY_NAMES = {
    'duration': 'Loan Duration (months)',
    'credit_amount': 'Credit Amount (DM)',
    'installment_rate': 'Installment Rate (% of income)',
    'residence_since': 'Residence Duration (years)',
    'age': 'Age (years)',
    'existing_credits': 'Existing Credits at Bank',
    'num_dependents': 'Number of Dependents',
    'checking_account': 'Checking Account Status',
    'credit_history': 'Credit History',
    'purpose': 'Loan Purpose',
    'savings_account': 'Savings Account Balance',
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


def get_human_readable_value(feature_name, code):
    """Convert a UCI code to a human-readable description."""
    mapping = FEATURE_VALUE_MAPPINGS.get(feature_name, {})
    return mapping.get(str(code), str(code))


def get_feature_display_name(feature_name):
    """Get human-readable display name for a feature."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' ').title())


def get_categorical_options(feature_name):
    """
    Get all valid options for a categorical feature.
    Returns list of (code, human_label) tuples.
    """
    mapping = FEATURE_VALUE_MAPPINGS.get(feature_name, {})
    # Filter out legacy A-codes to avoid duplicates — keep the OpenML-style keys
    seen_labels = set()
    options = []
    for code, label in mapping.items():
        if label not in seen_labels:
            options.append((code, label))
            seen_labels.add(label)
    return options


def format_applicant_for_display(applicant_dict):
    """
    Convert raw applicant dict to human-readable format for display.
    Returns dict of {display_name: display_value}.
    """
    result = {}
    for key, value in applicant_dict.items():
        display_name = get_feature_display_name(key)
        if key in FEATURE_VALUE_MAPPINGS:
            display_value = get_human_readable_value(key, value)
        else:
            display_value = value
        result[display_name] = display_value
    return result
