"""
CreditLens AI - Model Evaluation Module
=========================================
Evaluates both XGBoost and MLP on the held-out test set.
Calculates: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
Metrics are computed from actual model predictions — never hardcoded.
"""
import os
import sys
import json
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR


def evaluate_model(model, X_test, y_test, model_name='model'):
    """
    Evaluate a model on test data.
    
    Returns:
        metrics: dict with accuracy, precision, recall, f1, roc_auc
        cm: confusion matrix as list of lists
        roc_data: dict with fpr, tpr, thresholds
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'precision': round(float(precision_score(y_test, preds, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, preds, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, preds, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, probs)), 4),
    }
    
    cm = confusion_matrix(y_test, preds).tolist()
    
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_data = {
        'fpr': [round(float(x), 4) for x in fpr],
        'tpr': [round(float(x), 4) for x in tpr],
    }
    
    return metrics, cm, roc_data


def save_metrics(all_metrics):
    """Save all metrics to a JSON file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(all_metrics, f, indent=4)


def load_metrics():
    """Load metrics from the saved JSON file."""
    path = os.path.join(MODELS_DIR, 'metrics.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}. Run train.py first.")
    with open(path, 'r') as f:
        return json.load(f)


def format_metrics_for_display(metrics):
    """Format metrics dict for clean display."""
    return {
        'Accuracy': f"{metrics['accuracy'] * 100:.1f}%",
        'Precision': f"{metrics['precision'] * 100:.1f}%",
        'Recall': f"{metrics['recall'] * 100:.1f}%",
        'F1-Score': f"{metrics['f1']:.3f}",
        'ROC-AUC': f"{metrics['roc_auc']:.3f}",
    }
