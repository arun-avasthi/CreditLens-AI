"""
CreditLens AI - What-If Analysis Module
=========================================
Illustrative bounded sensitivity analysis:
"What happens if the loan amount or duration changes?"

IMPORTANT: This is an illustrative educational tool, NOT:
  - Financial advice
  - Guaranteed approval guidance
  - An optimal loan recommendation
  - A formal counterfactual explanation
"""
import os
import sys
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction import predict, load_model, load_preprocessor


def run_what_if_analysis(base_applicant, model_name='xgb_model',
                         amount_range=None, duration_range=None):
    """
    Run what-if analysis by varying credit_amount and/or duration.
    
    Args:
        base_applicant: dict of original applicant features
        model_name: which model to use
        amount_range: list of credit amounts to test (or None)
        duration_range: list of durations to test (or None)
        
    Returns:
        pd.DataFrame with columns: credit_amount, duration, approval_probability
    """
    model = load_model(model_name)
    preprocessor, _ = load_preprocessor()
    
    if amount_range is None:
        base_amt = float(base_applicant.get('credit_amount', 5000))
        amount_range = np.linspace(
            max(250, base_amt * 0.25),
            min(20000, base_amt * 3),
            10
        ).astype(int).tolist()
    
    if duration_range is None:
        base_dur = int(base_applicant.get('duration', 24))
        duration_range = list(range(
            max(6, base_dur - 18),
            min(72, base_dur + 24),
            6
        ))
    
    results = []
    
    for amt in amount_range:
        for dur in duration_range:
            applicant = copy.deepcopy(base_applicant)
            applicant['credit_amount'] = int(amt)
            applicant['duration'] = int(dur)
            
            pred = predict(applicant, model_name, model=model, preprocessor=preprocessor)
            results.append({
                'credit_amount': int(amt),
                'duration': int(dur),
                'approval_probability': pred['approval_probability'],
                'default_risk': pred['default_risk'],
                'risk_class': pred['risk_class'],
            })
    
    return pd.DataFrame(results)


def run_single_variable_what_if(base_applicant, variable, values, model_name='xgb_model'):
    """
    Run what-if analysis varying a single variable.
    
    Returns:
        pd.DataFrame with columns: variable, approval_probability, default_risk
    """
    model = load_model(model_name)
    preprocessor, _ = load_preprocessor()
    
    results = []
    for val in values:
        applicant = copy.deepcopy(base_applicant)
        applicant[variable] = val
        
        pred = predict(applicant, model_name, model=model, preprocessor=preprocessor)
        results.append({
            variable: val,
            'approval_probability': pred['approval_probability'],
            'default_risk': pred['default_risk'],
        })
    
    return pd.DataFrame(results)
