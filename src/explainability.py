"""
CreditLens AI - Explainability Module
=======================================
Provides SHAP (TreeExplainer for XGBoost) and LIME explanations.
Also includes SHAP vs LIME comparison functionality.

SHAP: Shows how individual features contribute to moving a prediction
      away from the model's baseline.

LIME: Creates a local linear approximation around a specific prediction.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LIME_NUM_SAMPLES, LIME_NUM_FEATURES, RANDOM_STATE


def get_shap_explainer(model, background_data=None):
    """
    Create a SHAP TreeExplainer for XGBoost.
    
    Args:
        model: trained XGBoost model
        background_data: optional background dataset for explainer
        
    Returns:
        shap.TreeExplainer
    """
    import shap
    # For XGBoost with enable_categorical, use tree_path_dependent
    # WITHOUT background data (background data triggers interventional mode
    # which is incompatible with categorical splits in current SHAP)
    try:
        explainer = shap.TreeExplainer(
            model, feature_perturbation='tree_path_dependent'
        )
    except Exception:
        # Fallback: try with background data
        try:
            if background_data is not None:
                explainer = shap.TreeExplainer(model, data=background_data)
            else:
                explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X_instance):
    """
    Compute SHAP values for a single instance.
    
    Args:
        explainer: SHAP TreeExplainer
        X_instance: 2D numpy array (1, n_features)
        
    Returns:
        shap_values: numpy array of SHAP values
        base_value: base value (expected value)
    """
    import shap
    
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    # Use the __call__ interface (works with categorical splits in newer XGBoost/SHAP)
    explanation = explainer(X_instance)
    
    # Extract values for the first (and only) instance
    sv = explanation.values
    if sv.ndim == 2:
        sv = sv[0]  # Single instance
    elif sv.ndim == 3:
        # Binary classification: shape (1, n_features, n_classes)
        sv = sv[0, :, 1] if sv.shape[2] > 1 else sv[0, :, 0]
    
    # Base value
    bv = explanation.base_values
    if isinstance(bv, np.ndarray):
        bv = bv.flatten()
        bv = float(bv[0]) if len(bv) == 1 else float(bv[1]) if len(bv) > 1 else 0.0
    else:
        bv = float(bv)
    
    return sv, float(bv)


def create_shap_waterfall_plot(shap_values, base_value, feature_names, instance_values=None, max_display=10):
    """
    Create a SHAP waterfall plot.
    
    Returns:
        matplotlib Figure
    """
    import shap
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=instance_values if instance_values is not None else np.zeros(len(shap_values)),
        feature_names=feature_names,
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def create_shap_bar_plot(shap_values, feature_names, top_n=10):
    """
    Create a horizontal bar chart of top SHAP feature contributions.
    
    Returns:
        matplotlib Figure
    """
    abs_vals = np.abs(shap_values)
    indices = np.argsort(abs_vals)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ef5350' if shap_values[i] < 0 else '#66bb6a' for i in indices]
    
    y_pos = range(len(indices))
    bars = ax.barh(y_pos, [shap_values[i] for i in indices], color=colors, edgecolor='none', height=0.6)
    
    ax.set_yticks(y_pos)
    labels = [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices]
    # Clean up feature names for display
    clean_labels = [_clean_feature_name(l) for l in labels]
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel('SHAP Value (impact on approval probability)', fontsize=10)
    ax.set_title('Top Feature Contributions (SHAP)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='#888888', linestyle='-', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def get_top_shap_features(shap_values, feature_names, top_n=5):
    """
    Get top N features by absolute SHAP value.
    
    Returns:
        list of dicts: [{feature, value, direction, abs_value}, ...]
    """
    abs_vals = np.abs(shap_values)
    indices = np.argsort(abs_vals)[-top_n:][::-1]
    
    results = []
    for i in indices:
        fname = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        results.append({
            'feature': _clean_feature_name(fname),
            'raw_feature': fname,
            'value': float(shap_values[i]),
            'direction': 'Increases approval' if shap_values[i] > 0 else 'Increases default risk',
            'abs_value': float(abs_vals[i]),
        })
    return results


def create_global_shap_importance(model, X_data, feature_names, top_n=15):
    """
    Compute global SHAP feature importance across multiple samples.
    
    Returns:
        matplotlib Figure, list of (feature_name, mean_abs_shap)
    """
    import shap
    
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    except Exception:
        explainer = shap.TreeExplainer(model)
    
    # Use __call__ interface for compatibility
    explanation = explainer(X_data)
    sv = explanation.values
    
    # Handle different output shapes
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    
    mean_abs = np.mean(np.abs(sv), axis=0)
    indices = np.argsort(mean_abs)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    y_pos = range(len(indices))
    ax.barh(y_pos, [mean_abs[i] for i in indices], color='#4FC3F7', edgecolor='none', height=0.6)
    
    labels = [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices]
    clean_labels = [_clean_feature_name(l) for l in labels]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
    ax.set_title('Global Feature Importance (SHAP)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    importance_list = [(feature_names[i], float(mean_abs[i])) for i in indices[::-1]]
    return fig, importance_list


def explain_lime(model, X_train, feature_names, instance, class_names=None):
    """
    Generate a LIME explanation for a single instance.
    
    Args:
        model: trained model with predict_proba
        X_train: training data (numpy array) for LIME explainer
        feature_names: list of feature names
        instance: 1D numpy array of the instance to explain
        class_names: optional class names
        
    Returns:
        lime explanation object
    """
    import lime.lime_tabular
    
    if class_names is None:
        class_names = ['Default Risk', 'Approval']
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(feature_names),
        class_names=class_names,
        mode='classification',
        random_state=RANDOM_STATE,
    )
    
    exp = explainer.explain_instance(
        data_row=np.array(instance).flatten(),
        predict_fn=model.predict_proba,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    
    return exp


def get_lime_features(lime_exp, class_idx=1, top_n=5):
    """
    Extract top features from LIME explanation.
    
    Returns:
        list of dicts: [{feature, value, direction}, ...]
    """
    exp_list = lime_exp.as_list(label=class_idx)
    
    results = []
    for feature_desc, weight in exp_list[:top_n]:
        results.append({
            'feature': _clean_feature_name(feature_desc),
            'raw_feature': feature_desc,
            'value': float(weight),
            'direction': 'Increases approval' if weight > 0 else 'Increases default risk',
            'abs_value': abs(float(weight)),
        })
    return results


def create_lime_plot(lime_exp, class_idx=1):
    """
    Create a bar chart of LIME feature contributions.
    
    Returns:
        matplotlib Figure
    """
    exp_list = lime_exp.as_list(label=class_idx)
    
    features = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#66bb6a' if w > 0 else '#ef5350' for w in weights]
    clean_features = [_clean_feature_name(f) for f in features]
    
    y_pos = range(len(clean_features))
    ax.barh(y_pos, weights, color=colors, edgecolor='none', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_features, fontsize=9)
    ax.set_xlabel('Feature Contribution (LIME)', fontsize=10)
    ax.set_title('Local Feature Contributions (LIME)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='#888888', linestyle='-', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def compare_shap_lime(shap_features, lime_features, top_n=5):
    """
    Descriptive comparison of SHAP and LIME top features.
    
    Args:
        shap_features: list from get_top_shap_features
        lime_features: list from get_lime_features
        
    Returns:
        dict with comparison data
    """
    shap_top = shap_features[:top_n]
    lime_top = lime_features[:top_n]
    
    shap_names = set(f['feature'] for f in shap_top)
    lime_names = set(f['feature'] for f in lime_top)
    
    shared = shap_names & lime_names
    
    # Direction agreement for shared features
    direction_matches = 0
    comparison_rows = []
    
    for sf in shap_top:
        lime_match = None
        for lf in lime_top:
            if sf['feature'] == lf['feature']:
                lime_match = lf
                break
        
        row = {
            'feature': sf['feature'],
            'shap_direction': sf['direction'],
            'lime_direction': lime_match['direction'] if lime_match else 'Not in LIME top',
            'directions_align': False,
        }
        
        if lime_match:
            row['directions_align'] = (sf['value'] > 0) == (lime_match['value'] > 0)
            if row['directions_align']:
                direction_matches += 1
        
        comparison_rows.append(row)
    
    return {
        'shared_features': len(shared),
        'total_compared': top_n,
        'direction_matches': direction_matches,
        'shap_only': shap_names - lime_names,
        'lime_only': lime_names - shap_names,
        'rows': comparison_rows,
    }


def _clean_feature_name(name):
    """Clean up a transformed feature name for display."""
    if name is None:
        return 'Unknown'
    name = str(name)
    # Remove common prefixes from ColumnTransformer
    for prefix in ['num__', 'cat__', 'num_', 'cat_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Clean up underscores
    name = name.replace('_', ' ')
    # Capitalize first letter
    if name:
        name = name[0].upper() + name[1:]
    return name
