"""
CreditLens AI - Stress Testing Module
========================================
One-variable-at-a-time sensitivity analysis.
Tests how model predictions change when a single variable varies
while all others remain constant.

This is NOT a Partial Dependence Plot.
"""
import os
import sys
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction import predict, load_model, load_preprocessor


# Default test ranges for each variable
DEFAULT_RANGES = {
    'credit_amount': np.linspace(250, 18500, 20).astype(int).tolist(),
    'duration': list(range(6, 73, 3)),
    'age': list(range(19, 76, 3)),
}


def run_stress_test(base_applicant, variable, values=None, model_name='xgb_model'):
    """
    One-variable-at-a-time sensitivity analysis.
    
    Args:
        base_applicant: dict of baseline applicant features
        variable: name of the variable to vary
        values: list of values to test (or None for defaults)
        model_name: which model to use
        
    Returns:
        pd.DataFrame with columns: {variable}, approval_probability, default_risk
    """
    if values is None:
        values = DEFAULT_RANGES.get(variable, [])
        if not values:
            raise ValueError(f"No default range defined for variable: {variable}")
    
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


def run_multi_stress_test(base_applicant, variables=None, model_name='xgb_model'):
    """
    Run stress tests for multiple variables.
    
    Returns:
        dict of {variable_name: pd.DataFrame}
    """
    if variables is None:
        variables = list(DEFAULT_RANGES.keys())
    
    results = {}
    for var in variables:
        results[var] = run_stress_test(base_applicant, var, model_name=model_name)
    
    return results
