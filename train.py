"""
CreditLens AI - Training Script
=================================
Trains XGBoost and MLP models, evaluates them, and saves all artifacts.

Run: python train.py

This script:
  1. Loads the UCI German Credit dataset
  2. Splits into 80/20 train/test (random_state=42)
  3. Builds and fits the preprocessing pipeline
  4. Trains XGBoost (n_estimators=150, max_depth=4, lr=0.05)
  5. Trains MLP (64-32 hidden, ReLU, Adam, early stopping)
  6. Evaluates both on the held-out test set
  7. Saves all artifacts to models/
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# Suppress convergence warnings during training
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    XGB_PARAMS, MLP_PARAMS, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, SHAP_BACKGROUND_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from src.data import load_data
from src.preprocessing import build_preprocessor, save_preprocessor, get_feature_names
from src.evaluation import evaluate_model, save_metrics


def main():
    print("=" * 60)
    print("  CreditLens AI — Model Training")
    print("=" * 60)
    print()
    
    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("[1/7] Loading UCI German Credit dataset...")
    df = load_data()
    print(f"  Dataset: {len(df)} records, {len(df.columns) - 1} features")
    print(f"  Target distribution: {dict(df['class'].value_counts())}")
    print(f"  (1=good/approval, 0=bad/default)")
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # ------------------------------------------------------------------
    # Step 2: Train-test split
    # ------------------------------------------------------------------
    print(f"\n[2/7] Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} records")
    print(f"  Test:  {len(X_test)} records")
    
    # ------------------------------------------------------------------
    # Step 3: Preprocessing
    # ------------------------------------------------------------------
    print("\n[3/7] Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)
    print(f"  Transformed features: {X_train_proc.shape[1]}")
    print(f"  ({len(NUMERIC_FEATURES)} numeric + {X_train_proc.shape[1] - len(NUMERIC_FEATURES)} one-hot encoded)")
    
    save_preprocessor(preprocessor, feature_names)
    print("  Preprocessor saved.")
    
    # ------------------------------------------------------------------
    # Step 4: Train XGBoost
    # ------------------------------------------------------------------
    print("\n[4/7] Training XGBoost...")
    xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != 'use_label_encoder'}
    xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False)
    xgb_model.fit(X_train_proc, y_train)
    print("  XGBoost training complete.")
    
    # ------------------------------------------------------------------
    # Step 5: Train MLP
    # ------------------------------------------------------------------
    print("\n[5/7] Training MLP (64→32, ReLU, Adam, early stopping)...")
    mlp_model = MLPClassifier(**MLP_PARAMS)
    mlp_model.fit(X_train_proc, y_train)
    n_iter = mlp_model.n_iter_
    print(f"  MLP training complete ({n_iter} iterations).")
    
    # ------------------------------------------------------------------
    # Step 6: Evaluate both models
    # ------------------------------------------------------------------
    print("\n[6/7] Evaluating models on held-out test set...")
    
    xgb_metrics, xgb_cm, xgb_roc = evaluate_model(xgb_model, X_test_proc, y_test, 'XGBoost')
    mlp_metrics, mlp_cm, mlp_roc = evaluate_model(mlp_model, X_test_proc, y_test, 'MLP')
    
    all_metrics = {
        'xgb': {
            'metrics': xgb_metrics,
            'confusion_matrix': xgb_cm,
            'roc': xgb_roc,
        },
        'mlp': {
            'metrics': mlp_metrics,
            'confusion_matrix': mlp_cm,
            'roc': mlp_roc,
        },
    }
    save_metrics(all_metrics)
    
    print("\n  XGBoost Metrics:")
    for k, v in xgb_metrics.items():
        print(f"    {k:>12s}: {v:.4f}")
    
    print("\n  MLP Metrics:")
    for k, v in mlp_metrics.items():
        print(f"    {k:>12s}: {v:.4f}")
    
    # ------------------------------------------------------------------
    # Step 7: Save all artifacts
    # ------------------------------------------------------------------
    print("\n[7/7] Saving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.joblib'))
    joblib.dump(mlp_model, os.path.join(MODELS_DIR, 'mlp_model.joblib'))
    
    # Save background samples for SHAP
    bg_size = min(SHAP_BACKGROUND_SIZE, len(X_train_proc))
    np.random.seed(RANDOM_STATE)
    bg_indices = np.random.choice(len(X_train_proc), bg_size, replace=False)
    background_samples = X_train_proc[bg_indices]
    joblib.dump(background_samples, os.path.join(MODELS_DIR, 'background_samples.joblib'))
    
    # Save test samples for LIME background
    test_size = min(SHAP_BACKGROUND_SIZE, len(X_test_proc))
    joblib.dump(X_train_proc[:test_size], os.path.join(MODELS_DIR, 'test_samples.joblib'))
    
    # Save manifest
    manifest = {
        'models': ['xgb_model.joblib', 'mlp_model.joblib'],
        'preprocessor': 'preprocessor.joblib',
        'feature_names': 'feature_names.joblib',
        'background_samples': 'background_samples.joblib',
        'test_samples': 'test_samples.joblib',
        'metrics': 'metrics.json',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': int(X_train_proc.shape[1]),
        'random_state': RANDOM_STATE,
    }
    with open(os.path.join(MODELS_DIR, 'model_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print("  All artifacts saved to models/")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Models:      XGBoost, MLP")
    print(f"  Features:    {X_train_proc.shape[1]} transformed")
    print(f"  Train/Test:  {len(X_train)}/{len(X_test)}")
    print(f"  XGB ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    print(f"  MLP ROC-AUC: {mlp_metrics['roc_auc']:.4f}")
    print(f"  Artifacts:   {MODELS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
