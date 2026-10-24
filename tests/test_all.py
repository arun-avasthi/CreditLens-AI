"""
CreditLens AI - Test Suite
============================
Comprehensive tests for all modules.

Run: python -m pytest tests/test_all.py -v
     python -m unittest discover -s tests -v
"""
import os
import sys
import json
import unittest
import warnings
import tempfile
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore')

from config import (
    MODELS_DIR, THRESHOLD, RANDOM_STATE, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, ALL_FEATURES, XGB_PARAMS, MLP_PARAMS
)


def _get_sample_applicant():
    """Return a valid sample applicant dict."""
    return {
        'duration': 24,
        'credit_amount': 5000,
        'installment_rate': 2,
        'residence_since': 3,
        'age': 35,
        'existing_credits': 1,
        'num_dependents': 1,
        'checking_account': "'>=200'",
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


def _get_high_risk_applicant():
    """Return a high-risk sample applicant dict."""
    return {
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


# ==============================================================================
# Test Data Module
# ==============================================================================
class TestData(unittest.TestCase):
    """Test dataset loading and validation."""

    def test_load_data_returns_dataframe(self):
        try:
            from src.data import load_data
            df = load_data()
            self.assertIsInstance(df, pd.DataFrame)
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_dataset_has_1000_records(self):
        try:
            from src.data import load_data
            df = load_data()
            self.assertEqual(len(df), 1000)
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_dataset_has_correct_columns(self):
        try:
            from src.data import load_data
            df = load_data()
            expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['class'])
            self.assertEqual(set(df.columns), expected)
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_target_is_binary(self):
        try:
            from src.data import load_data
            df = load_data()
            self.assertTrue(df['class'].isin([0, 1]).all())
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_target_distribution(self):
        try:
            from src.data import load_data
            df = load_data()
            self.assertEqual(df['class'].sum(), 700)  # 700 good credits
            self.assertEqual((df['class'] == 0).sum(), 300)  # 300 bad credits
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_numeric_features_are_numeric(self):
        try:
            from src.data import load_data
            df = load_data()
            for col in NUMERIC_FEATURES:
                self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric")
        except (RuntimeError, PermissionError, OSError) as e:
            self.skipTest(f"Dataset not accessible (network/permission): {e}")

    def test_dataset_info(self):
        from src.data import get_dataset_info
        info = get_dataset_info()
        self.assertEqual(info['records'], 1000)
        self.assertEqual(info['numeric_features'], 7)
        self.assertEqual(info['categorical_features'], 13)


# ==============================================================================
# Test Configuration
# ==============================================================================
class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_feature_lists_complete(self):
        self.assertEqual(len(NUMERIC_FEATURES), 7)
        self.assertEqual(len(CATEGORICAL_FEATURES), 13)
        self.assertEqual(len(ALL_FEATURES), 20)

    def test_threshold_valid(self):
        self.assertGreater(THRESHOLD, 0)
        self.assertLess(THRESHOLD, 1)

    def test_models_dir_exists(self):
        self.assertTrue(os.path.isdir(MODELS_DIR))

    def test_random_state_set(self):
        self.assertEqual(RANDOM_STATE, 42)

    def test_xgb_params_valid(self):
        self.assertIn('n_estimators', XGB_PARAMS)
        self.assertIn('max_depth', XGB_PARAMS)
        self.assertGreater(XGB_PARAMS['n_estimators'], 0)

    def test_mlp_params_valid(self):
        self.assertIn('hidden_layer_sizes', MLP_PARAMS)
        self.assertIn('activation', MLP_PARAMS)


# ==============================================================================
# Test Preprocessing Module
# ==============================================================================
class TestPreprocessing(unittest.TestCase):
    """Test preprocessing pipeline."""

    def test_preprocessor_loads(self):
        from src.preprocessing import load_preprocessor
        preprocessor, feature_names = load_preprocessor()
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(feature_names)

    def test_feature_count_approximately_61(self):
        from src.preprocessing import load_preprocessor
        _, feature_names = load_preprocessor()
        self.assertEqual(len(feature_names), 61)

    def test_preprocessor_transform_single_sample(self):
        from src.preprocessing import load_preprocessor
        preprocessor, feature_names = load_preprocessor()

        sample = {col: 0 for col in NUMERIC_FEATURES}
        sample.update({col: 'none' for col in CATEGORICAL_FEATURES})
        df = pd.DataFrame([sample])
        for c in NUMERIC_FEATURES:
            df[c] = pd.to_numeric(df[c])
        for c in CATEGORICAL_FEATURES:
            df[c] = df[c].astype(str)

        X = preprocessor.transform(df)
        self.assertEqual(X.shape[1], 61)

    def test_preprocess_applicant_function(self):
        from src.preprocessing import preprocess_applicant
        applicant = _get_sample_applicant()
        X = preprocess_applicant(applicant)
        self.assertEqual(X.shape, (1, 61))

    def test_preprocess_applicant_missing_feature_raises(self):
        from src.preprocessing import preprocess_applicant
        incomplete = {'duration': 24}  # Missing most features
        with self.assertRaises(ValueError):
            preprocess_applicant(incomplete)


# ==============================================================================
# Test Prediction Module
# ==============================================================================
class TestPrediction(unittest.TestCase):
    """Test prediction engine."""

    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        import joblib
        cls.xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.joblib'))
        cls.mlp_model = joblib.load(os.path.join(MODELS_DIR, 'mlp_model.joblib'))

    def test_xgb_prediction_returns_dict(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        self.assertIsInstance(result, dict)

    def test_mlp_prediction_returns_dict(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'mlp_model')
        self.assertIsInstance(result, dict)

    def test_prediction_has_required_keys(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        required_keys = ['approval_probability', 'default_risk', 'risk_class', 'approved', 'threshold']
        for key in required_keys:
            self.assertIn(key, result)

    def test_probability_range(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        self.assertGreaterEqual(result['approval_probability'], 0.0)
        self.assertLessEqual(result['approval_probability'], 1.0)
        self.assertGreaterEqual(result['default_risk'], 0.0)
        self.assertLessEqual(result['default_risk'], 1.0)

    def test_probabilities_sum_to_one(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        total = result['approval_probability'] + result['default_risk']
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_threshold_logic(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        if result['approval_probability'] >= THRESHOLD:
            self.assertTrue(result['approved'])
        else:
            self.assertFalse(result['approved'])

    def test_risk_class_values(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'xgb_model')
        self.assertIn(result['risk_class'], ['Low Risk', 'Medium Risk', 'High Risk'])

    def test_low_risk_profile_gives_higher_approval(self):
        """Low-risk applicant should have higher approval probability than high-risk."""
        from src.prediction import predict
        low = predict(_get_sample_applicant(), 'xgb_model')
        high = predict(_get_high_risk_applicant(), 'xgb_model')
        self.assertGreater(low['approval_probability'], high['approval_probability'])

    def test_mlp_probability_range(self):
        from src.prediction import predict
        result = predict(_get_sample_applicant(), 'mlp_model')
        self.assertGreaterEqual(result['approval_probability'], 0.0)
        self.assertLessEqual(result['approval_probability'], 1.0)
        total = result['approval_probability'] + result['default_risk']
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_batch_prediction(self):
        from src.prediction import predict_batch
        results = predict_batch([_get_sample_applicant(), _get_high_risk_applicant()])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn('approval_probability', r)

    def test_model_classes_are_binary(self):
        """Verify model class order: [0, 1]."""
        np.testing.assert_array_equal(self.xgb_model.classes_, [0, 1])
        np.testing.assert_array_equal(self.mlp_model.classes_, [0, 1])


# ==============================================================================
# Test Evaluation Module
# ==============================================================================
class TestEvaluation(unittest.TestCase):
    """Test model evaluation and metrics."""

    def test_metrics_file_exists(self):
        path = os.path.join(MODELS_DIR, 'metrics.json')
        self.assertTrue(os.path.exists(path))

    def test_metrics_valid_json(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        self.assertIsInstance(metrics, dict)

    def test_metrics_has_both_models(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        self.assertIn('xgb', metrics)
        self.assertIn('mlp', metrics)

    def test_metrics_has_required_fields(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        required = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for model_key in ['xgb', 'mlp']:
            for field in required:
                self.assertIn(field, metrics[model_key]['metrics'])

    def test_metrics_in_valid_range(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        for model_key in ['xgb', 'mlp']:
            for field in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                val = metrics[model_key]['metrics'][field]
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_confusion_matrix_exists(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        for model_key in ['xgb', 'mlp']:
            self.assertIn('confusion_matrix', metrics[model_key])
            cm = metrics[model_key]['confusion_matrix']
            self.assertEqual(len(cm), 2)
            self.assertEqual(len(cm[0]), 2)

    def test_roc_data_exists(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        for model_key in ['xgb', 'mlp']:
            self.assertIn('roc', metrics[model_key])
            roc = metrics[model_key]['roc']
            self.assertIn('fpr', roc)
            self.assertIn('tpr', roc)

    def test_confusion_matrix_sums_to_test_size(self):
        from src.evaluation import load_metrics
        metrics = load_metrics()
        for model_key in ['xgb', 'mlp']:
            cm = metrics[model_key]['confusion_matrix']
            total = sum(sum(row) for row in cm)
            self.assertEqual(total, 200, f"{model_key} CM should sum to 200 test samples")

    def test_format_metrics_for_display(self):
        from src.evaluation import format_metrics_for_display
        sample = {'accuracy': 0.72, 'precision': 0.75, 'recall': 0.83, 'f1': 0.79, 'roc_auc': 0.76}
        result = format_metrics_for_display(sample)
        self.assertIn('Accuracy', result)
        self.assertIn('%', result['Accuracy'])


# ==============================================================================
# Test Explainability Module
# ==============================================================================
class TestExplainability(unittest.TestCase):
    """Test SHAP and LIME explainability."""

    @classmethod
    def setUpClass(cls):
        """Load models and preprocessor once."""
        import joblib
        cls.xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.joblib'))
        cls.mlp_model = joblib.load(os.path.join(MODELS_DIR, 'mlp_model.joblib'))
        cls.preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        cls.feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
        cls.bg_samples = joblib.load(os.path.join(MODELS_DIR, 'background_samples.joblib'))
        cls.test_samples = joblib.load(os.path.join(MODELS_DIR, 'test_samples.joblib'))

        # Create a test instance
        from src.preprocessing import preprocess_applicant
        cls.test_instance = preprocess_applicant(_get_sample_applicant(), cls.preprocessor)

    def test_shap_explainer_creation(self):
        from src.explainability import get_shap_explainer
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        self.assertIsNotNone(explainer)

    def test_shap_values_computation(self):
        from src.explainability import get_shap_explainer, compute_shap_values
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, base_val = compute_shap_values(explainer, self.test_instance)
        self.assertEqual(len(shap_vals), 61)
        self.assertIsInstance(base_val, float)

    def test_shap_values_are_finite(self):
        from src.explainability import get_shap_explainer, compute_shap_values
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, base_val = compute_shap_values(explainer, self.test_instance)
        self.assertTrue(np.all(np.isfinite(shap_vals)))
        self.assertTrue(np.isfinite(base_val))

    def test_shap_top_features(self):
        from src.explainability import get_shap_explainer, compute_shap_values, get_top_shap_features
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, _ = compute_shap_values(explainer, self.test_instance)
        top = get_top_shap_features(shap_vals, self.feature_names, top_n=5)
        self.assertEqual(len(top), 5)
        self.assertIn('feature', top[0])
        self.assertIn('direction', top[0])
        self.assertIn('value', top[0])
        self.assertIn('abs_value', top[0])

    def test_shap_bar_plot(self):
        from src.explainability import get_shap_explainer, compute_shap_values, create_shap_bar_plot
        import matplotlib
        matplotlib.use('Agg')
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, _ = compute_shap_values(explainer, self.test_instance)
        fig = create_shap_bar_plot(shap_vals, self.feature_names)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_shap_waterfall_plot(self):
        from src.explainability import get_shap_explainer, compute_shap_values, create_shap_waterfall_plot
        import matplotlib
        matplotlib.use('Agg')
        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, base_val = compute_shap_values(explainer, self.test_instance)
        fig = create_shap_waterfall_plot(
            shap_vals, base_val, self.feature_names,
            instance_values=self.test_instance.flatten(),
            max_display=10
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_lime_explanation_xgboost(self):
        from src.explainability import explain_lime
        lime_exp = explain_lime(
            self.xgb_model, self.test_samples, self.feature_names,
            self.test_instance.flatten()
        )
        self.assertIsNotNone(lime_exp)

    def test_lime_explanation_mlp(self):
        """LIME should also work with MLP model."""
        from src.explainability import explain_lime
        lime_exp = explain_lime(
            self.mlp_model, self.test_samples, self.feature_names,
            self.test_instance.flatten()
        )
        self.assertIsNotNone(lime_exp)

    def test_lime_features(self):
        from src.explainability import explain_lime, get_lime_features
        lime_exp = explain_lime(
            self.xgb_model, self.test_samples, self.feature_names,
            self.test_instance.flatten()
        )
        features = get_lime_features(lime_exp, top_n=5)
        self.assertEqual(len(features), 5)
        self.assertIn('feature', features[0])
        self.assertIn('value', features[0])

    def test_lime_plot(self):
        from src.explainability import explain_lime, create_lime_plot
        import matplotlib
        matplotlib.use('Agg')
        lime_exp = explain_lime(
            self.xgb_model, self.test_samples, self.feature_names,
            self.test_instance.flatten()
        )
        fig = create_lime_plot(lime_exp)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_shap_lime_comparison(self):
        from src.explainability import (
            get_shap_explainer, compute_shap_values, get_top_shap_features,
            explain_lime, get_lime_features, compare_shap_lime
        )

        explainer = get_shap_explainer(self.xgb_model, self.bg_samples)
        shap_vals, _ = compute_shap_values(explainer, self.test_instance)
        shap_feats = get_top_shap_features(shap_vals, self.feature_names, top_n=5)

        lime_exp = explain_lime(
            self.xgb_model, self.test_samples, self.feature_names,
            self.test_instance.flatten()
        )
        lime_feats = get_lime_features(lime_exp, top_n=5)

        comparison = compare_shap_lime(shap_feats, lime_feats)
        self.assertIn('shared_features', comparison)
        self.assertIn('rows', comparison)
        self.assertIn('total_compared', comparison)
        self.assertIsInstance(comparison['shared_features'], int)
        self.assertGreaterEqual(comparison['shared_features'], 0)

    def test_clean_feature_name(self):
        from src.explainability import _clean_feature_name
        self.assertEqual(_clean_feature_name('num__age'), 'Age')
        self.assertEqual(_clean_feature_name('cat__housing_own'), 'Housing own')
        self.assertEqual(_clean_feature_name(None), 'Unknown')

    def test_global_shap_importance(self):
        from src.explainability import create_global_shap_importance
        import matplotlib
        matplotlib.use('Agg')
        fig, importance_list = create_global_shap_importance(
            self.xgb_model, self.bg_samples, self.feature_names, top_n=5
        )
        self.assertIsNotNone(fig)
        self.assertGreater(len(importance_list), 0)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ==============================================================================
# Test What-If Analysis
# ==============================================================================
class TestWhatIf(unittest.TestCase):
    """Test what-if analysis."""

    def test_what_if_returns_dataframe(self):
        from src.what_if import run_single_variable_what_if
        result = run_single_variable_what_if(_get_sample_applicant(), 'credit_amount', [2000, 5000, 10000])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_what_if_has_probability_column(self):
        from src.what_if import run_single_variable_what_if
        result = run_single_variable_what_if(_get_sample_applicant(), 'duration', [12, 24, 36])
        self.assertIn('approval_probability', result.columns)

    def test_what_if_probabilities_valid(self):
        from src.what_if import run_single_variable_what_if
        result = run_single_variable_what_if(_get_sample_applicant(), 'credit_amount', [1000, 5000, 15000])
        self.assertTrue((result['approval_probability'] >= 0).all())
        self.assertTrue((result['approval_probability'] <= 1).all())

    def test_what_if_amount_affects_probability(self):
        """Changing credit amount should generally affect the probability."""
        from src.what_if import run_single_variable_what_if
        result = run_single_variable_what_if(_get_sample_applicant(), 'credit_amount',
                                             [500, 2000, 5000, 10000, 18000])
        # Probabilities should not all be identical
        self.assertGreater(result['approval_probability'].std(), 0.001)

    def test_what_if_with_mlp(self):
        from src.what_if import run_single_variable_what_if
        result = run_single_variable_what_if(_get_sample_applicant(), 'duration', [12, 24],
                                             model_name='mlp_model')
        self.assertEqual(len(result), 2)


# ==============================================================================
# Test Stress Testing
# ==============================================================================
class TestStressTesting(unittest.TestCase):
    """Test stress testing / sensitivity analysis."""

    def test_stress_test_credit_amount(self):
        from src.stress_testing import run_stress_test
        result = run_stress_test(_get_sample_applicant(), 'credit_amount')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_stress_test_duration(self):
        from src.stress_testing import run_stress_test
        result = run_stress_test(_get_sample_applicant(), 'duration')
        self.assertIsInstance(result, pd.DataFrame)

    def test_stress_test_age(self):
        from src.stress_testing import run_stress_test
        result = run_stress_test(_get_sample_applicant(), 'age')
        self.assertIsInstance(result, pd.DataFrame)

    def test_stress_test_has_valid_probabilities(self):
        from src.stress_testing import run_stress_test
        result = run_stress_test(_get_sample_applicant(), 'credit_amount')
        self.assertTrue((result['approval_probability'] >= 0).all())
        self.assertTrue((result['approval_probability'] <= 1).all())

    def test_default_ranges_exist(self):
        from src.stress_testing import DEFAULT_RANGES
        self.assertIn('credit_amount', DEFAULT_RANGES)
        self.assertIn('duration', DEFAULT_RANGES)
        self.assertIn('age', DEFAULT_RANGES)


# ==============================================================================
# Test PDF Report
# ==============================================================================
class TestPDFReport(unittest.TestCase):
    """Test PDF report generation."""

    def test_generate_pdf_returns_bytes(self):
        from reports.pdf_report import generate_pdf_report
        result = generate_pdf_report(
            applicant_data={'Age': 35, 'Credit Amount': 5000},
            prediction_results={
                'approval_probability': 0.65,
                'default_risk': 0.35,
                'risk_class': 'Low Risk',
                'approved': True,
            },
        )
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 100)

    def test_pdf_with_features(self):
        from reports.pdf_report import generate_pdf_report
        result = generate_pdf_report(
            applicant_data={'Age': 35},
            prediction_results={
                'approval_probability': 0.65,
                'default_risk': 0.35,
                'risk_class': 'Low Risk',
                'approved': True,
            },
            top_features=[
                {'feature': 'Age', 'direction': 'Increases approval', 'value': 0.05},
            ],
            metrics={'accuracy': 0.72, 'roc_auc': 0.76},
        )
        self.assertIsInstance(result, bytes)

    def test_pdf_starts_with_pdf_header(self):
        from reports.pdf_report import generate_pdf_report
        result = generate_pdf_report(
            applicant_data={},
            prediction_results={
                'approval_probability': 0.5,
                'default_risk': 0.5,
                'risk_class': 'Medium Risk',
                'approved': True,
            },
        )
        self.assertTrue(result[:4] == b'%PDF')

    def test_pdf_with_unicode_characters(self):
        """PDF should handle Unicode input without crashing."""
        from reports.pdf_report import generate_pdf_report
        result = generate_pdf_report(
            applicant_data={'Feature': '≥ 200 DM', 'Status': '< 0 DM'},
            prediction_results={
                'approval_probability': 0.7,
                'default_risk': 0.3,
                'risk_class': 'Low Risk',
                'approved': True,
            },
        )
        self.assertTrue(result[:4] == b'%PDF')

    def test_pdf_with_many_features(self):
        """PDF should handle all 20 applicant features."""
        from reports.pdf_report import generate_pdf_report
        from src.feature_mapping import format_applicant_for_display
        display_data = format_applicant_for_display(_get_sample_applicant())
        result = generate_pdf_report(
            applicant_data=display_data,
            prediction_results={
                'approval_probability': 0.65,
                'default_risk': 0.35,
                'risk_class': 'Low Risk',
                'approved': True,
            },
            top_features=[
                {'feature': f'Feature {i}', 'direction': 'Increases approval', 'value': 0.01 * i}
                for i in range(7)
            ],
            metrics={'accuracy': 0.72, 'precision': 0.77, 'recall': 0.83, 'f1': 0.80, 'roc_auc': 0.76},
            model_name='XGBoost',
        )
        self.assertGreater(len(result), 500)
        self.assertTrue(result[:4] == b'%PDF')

    def test_pdf_high_risk_report(self):
        from reports.pdf_report import generate_pdf_report
        result = generate_pdf_report(
            applicant_data={'Age': 23, 'Credit Amount': 12000},
            prediction_results={
                'approval_probability': 0.3,
                'default_risk': 0.7,
                'risk_class': 'High Risk',
                'approved': False,
            },
        )
        self.assertTrue(result[:4] == b'%PDF')


# ==============================================================================
# Test Feature Mapping
# ==============================================================================
class TestFeatureMapping(unittest.TestCase):
    """Test feature mapping functions."""

    def test_get_human_readable_value(self):
        from src.feature_mapping import get_human_readable_value
        result = get_human_readable_value('housing', 'own')
        self.assertEqual(result, 'Own')

    def test_get_feature_display_name(self):
        from src.feature_mapping import get_feature_display_name
        result = get_feature_display_name('credit_amount')
        self.assertEqual(result, 'Credit Amount (DM)')

    def test_format_applicant_for_display(self):
        from src.feature_mapping import format_applicant_for_display
        result = format_applicant_for_display({
            'duration': 24,
            'housing': 'own',
        })
        self.assertIn('Loan Duration (months)', result)
        self.assertIn('Housing', result)

    def test_all_features_have_display_names(self):
        from src.feature_mapping import FEATURE_DISPLAY_NAMES
        for feature in ALL_FEATURES:
            self.assertIn(feature, FEATURE_DISPLAY_NAMES)

    def test_all_categorical_features_have_mappings(self):
        from src.feature_mapping import FEATURE_VALUE_MAPPINGS
        for feature in CATEGORICAL_FEATURES:
            self.assertIn(feature, FEATURE_VALUE_MAPPINGS)
            self.assertGreater(len(FEATURE_VALUE_MAPPINGS[feature]), 0)

    def test_unknown_feature_returns_code(self):
        from src.feature_mapping import get_human_readable_value
        result = get_human_readable_value('housing', 'UNKNOWN_CODE')
        self.assertEqual(result, 'UNKNOWN_CODE')

    def test_get_categorical_options(self):
        from src.feature_mapping import get_categorical_options
        options = get_categorical_options('housing')
        self.assertGreater(len(options), 0)
        # Should be list of (code, label) tuples
        for code, label in options:
            self.assertIsInstance(label, str)


# ==============================================================================
# Test Model Artifacts
# ==============================================================================
class TestArtifacts(unittest.TestCase):
    """Test model artifact integrity."""

    def test_xgb_model_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'xgb_model.joblib')))

    def test_mlp_model_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'mlp_model.joblib')))

    def test_preprocessor_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'preprocessor.joblib')))

    def test_feature_names_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'feature_names.joblib')))

    def test_background_samples_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'background_samples.joblib')))

    def test_metrics_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'metrics.json')))

    def test_manifest_exists(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, 'model_manifest.json')))

    def test_manifest_valid(self):
        with open(os.path.join(MODELS_DIR, 'model_manifest.json'), 'r') as f:
            manifest = json.load(f)
        self.assertIn('models', manifest)
        self.assertIn('n_features', manifest)
        self.assertEqual(manifest['n_features'], 61)

    def test_background_samples_shape(self):
        import joblib
        bg = joblib.load(os.path.join(MODELS_DIR, 'background_samples.joblib'))
        self.assertEqual(bg.shape[1], 61)
        self.assertGreater(bg.shape[0], 0)

    def test_test_samples_shape(self):
        import joblib
        ts = joblib.load(os.path.join(MODELS_DIR, 'test_samples.joblib'))
        self.assertEqual(ts.shape[1], 61)

    def test_feature_names_count(self):
        import joblib
        names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
        self.assertEqual(len(names), 61)


# ==============================================================================
# Test Edge Cases
# ==============================================================================
class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_prediction_with_unknown_category(self):
        """Model should handle unknown categorical values via handle_unknown='ignore'."""
        from src.prediction import predict
        applicant = _get_sample_applicant()
        applicant['checking_account'] = 'UNKNOWN_VALUE'
        result = predict(applicant, 'xgb_model')
        self.assertIsInstance(result, dict)
        self.assertIn('approval_probability', result)

    def test_extreme_credit_amount(self):
        from src.prediction import predict
        applicant = _get_sample_applicant()
        applicant['credit_amount'] = 100000
        result = predict(applicant, 'xgb_model')
        self.assertGreaterEqual(result['approval_probability'], 0.0)
        self.assertLessEqual(result['approval_probability'], 1.0)

    def test_extreme_age(self):
        from src.prediction import predict
        applicant = _get_sample_applicant()
        applicant['age'] = 18
        result = predict(applicant, 'xgb_model')
        self.assertIn('approval_probability', result)

    def test_missing_model_raises(self):
        from src.prediction import predict
        with self.assertRaises(FileNotFoundError):
            predict(_get_sample_applicant(), 'nonexistent_model')

    def test_boundary_installment_rate(self):
        from src.prediction import predict
        for rate in [1, 4]:
            applicant = _get_sample_applicant()
            applicant['installment_rate'] = rate
            result = predict(applicant, 'xgb_model')
            self.assertGreaterEqual(result['approval_probability'], 0.0)
            self.assertLessEqual(result['approval_probability'], 1.0)

    def test_all_unknown_categories(self):
        """All unknown categories should not crash."""
        from src.prediction import predict
        applicant = _get_sample_applicant()
        for cat in CATEGORICAL_FEATURES:
            applicant[cat] = 'COMPLETELY_UNKNOWN'
        result = predict(applicant, 'xgb_model')
        self.assertIn('approval_probability', result)


# ==============================================================================
# Test Streamlit App Imports
# ==============================================================================
class TestAppImports(unittest.TestCase):
    """Test that the app module can be imported without errors."""

    def test_app_file_exists(self):
        app_path = os.path.join(PROJECT_ROOT, 'app.py')
        self.assertTrue(os.path.exists(app_path))

    def test_app_syntax_valid(self):
        """Verify app.py has valid Python syntax."""
        app_path = os.path.join(PROJECT_ROOT, 'app.py')
        with open(app_path, 'r') as f:
            source = f.read()
        compile(source, app_path, 'exec')

    def test_train_syntax_valid(self):
        """Verify train.py has valid Python syntax."""
        train_path = os.path.join(PROJECT_ROOT, 'train.py')
        with open(train_path, 'r') as f:
            source = f.read()
        compile(source, train_path, 'exec')

    def test_run_sh_exists(self):
        run_path = os.path.join(PROJECT_ROOT, 'run.sh')
        self.assertTrue(os.path.exists(run_path))


if __name__ == '__main__':
    unittest.main()
