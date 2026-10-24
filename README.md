# 🔍 CreditLens AI

**Explainable Credit Risk Analysis Dashboard**

A professional, end-to-end machine learning application that predicts credit risk and explains model decisions using XGBoost, MLP, SHAP, and LIME. Built with Python and Streamlit.

> ⚠️ **Educational Prototype** — This application uses a historical German credit dataset (1994, 1,000 records) and should not be used for real lending decisions.

Link:https://creditlens-aigit-p8wwi2pa9mv9vqjypxuyu4.streamlit.app/
---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Credit Risk Prediction** | XGBoost and MLP predictions with probability estimates |
| **SHAP Explainability** | Feature contribution analysis with waterfall and bar plots |
| **LIME Explainability** | Model-agnostic local explanations for any model |
| **SHAP vs LIME Comparison** | Side-by-side method comparison with agreement analysis |
| **What-if Scenario Analysis** | Explore how input changes affect predictions |
| **Stress Testing** | Sensitivity analysis across feature ranges with model comparison |
| **Global Feature Importance** | Dataset-level SHAP + XGBoost built-in importance |
| **Model Performance** | Confusion matrices, ROC curves, precision-recall, benchmarks |
| **PDF Reports** | Professional downloadable PDF analysis reports |
| **Fairness & Responsible AI** | Limitations, privacy, and responsible use documentation |
| **CSV Data Export** | Export prediction data for further analysis |
| **Demo Profiles** | Low, moderate, and high-risk demonstration profiles |

---

## 🖥️ Screenshots

After launching the application, navigate through the sidebar to explore:

1. **Overview Dashboard** — Key metrics, model performance summary, and workflow guide
2. **Credit Risk Prediction** — Input form with demo profiles, probability bars, and quick SHAP explanations
3. **SHAP / LIME Explainability** — Detailed feature contribution analysis
4. **What-if Analysis** — Interactive scenario exploration with sensitivity charts
5. **Model Performance** — Confusion matrices, ROC curves, and metric comparisons

---

## 🏗️ Architecture

```
CreditLens AI/
├── app.py                    # Main Streamlit application (12 pages)
├── config.py                 # Central configuration and constants
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── run.sh                    # Launcher script (macOS/Linux)
├── .streamlit/config.toml    # Streamlit dark theme configuration
├── src/                      # Backend modules
│   ├── data.py              # Dataset loading (UCI German Credit via OpenML)
│   ├── preprocessing.py     # StandardScaler + OneHotEncoder pipeline
│   ├── prediction.py        # Prediction engine
│   ├── evaluation.py        # Model evaluation metrics
│   ├── explainability.py    # SHAP + LIME explanations
│   ├── feature_mapping.py   # UCI code → human-readable labels
│   ├── what_if.py           # What-if scenario analysis
│   └── stress_testing.py    # Sensitivity analysis
├── models/                   # Saved model artifacts (pre-trained)
│   ├── xgb_model.joblib     # Trained XGBoost classifier
│   ├── mlp_model.joblib     # Trained MLP classifier
│   ├── preprocessor.joblib  # Fitted ColumnTransformer
│   ├── feature_names.joblib # 61 transformed feature names
│   ├── background_samples.joblib  # SHAP background data (100 samples)
│   ├── test_samples.joblib  # LIME training data (100 samples)
│   ├── metrics.json         # Held-out test set evaluation metrics
│   └── model_manifest.json  # Artifact metadata
├── reports/                  # Report generation
│   └── pdf_report.py        # PDF report generator (fpdf2)
└── tests/                    # Test suite
    └── test_all.py          # Comprehensive unit and integration tests
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (tested with Python 3.13)
- macOS, Linux, or Windows
- ~500 MB disk space (for dependencies)

### Option 1: Using the launcher script (recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CreditLens-AI.git
cd CreditLens-AI

# Make the launcher executable and run
chmod +x run.sh
./run.sh
```

The launcher will:
1. Check your Python installation
2. Create a virtual environment (`.venv/`)
3. Install all dependencies
4. Verify model artifacts (train if missing)
5. Start Streamlit on `http://localhost:8501`

### Option 2: Manual setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CreditLens-AI.git
cd CreditLens-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models — pre-trained artifacts are included
# python train.py

# Start the application
streamlit run app.py
```

### Open the dashboard

Navigate to **http://localhost:8501** in your browser.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | UCI Statlog German Credit |
| **Source** | UCI Machine Learning Repository (OpenML ID: 31) |
| **Records** | 1,000 |
| **Features** | 20 (7 numeric, 13 categorical) |
| **Target** | Binary — good credit (1) / bad credit (0) |
| **Year** | 1994 |
| **Currency** | Deutsche Mark (DM) |

---

## 🤖 Models

### XGBoost (Primary Model)
- Gradient boosting classifier
- 150 estimators, max depth 4, learning rate 0.05
- SHAP TreeExplainer for feature attribution

### MLP (Comparison Model)
- Neural network (64 → 32 hidden units)
- ReLU activation, Adam optimizer, early stopping
- LIME for model-agnostic explanations

### Evaluation (Held-out Test Set — 200 records)

| Metric | XGBoost | MLP |
|--------|---------|-----|
| Accuracy | 71.5% | 69.5% |
| Precision | 77.5% | 70.1% |
| Recall | 83.6% | 98.6% |
| F1-Score | 0.804 | 0.819 |
| ROC-AUC | 0.761 | 0.709 |

---

## 🧪 Testing

```bash
# Run all tests
.venv/bin/python -m pytest tests/test_all.py -v

# Run with unittest
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover:
- Dataset loading and validation
- Preprocessing pipeline (61 transformed features)
- Prediction engine (both models, probability ranges, class mappings)
- SHAP explainability (explainer, values, plots)
- LIME explainability (explanations, feature extraction)
- SHAP vs LIME comparison
- What-if analysis
- Stress testing
- PDF report generation and validity
- Feature mapping and display names
- Model artifacts integrity
- Edge cases (unknown categories, extreme values)

---

## 🚀 Deployment

### Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

> **Note:** Model artifacts (`.joblib` files in `models/`) must be included in the repository. Total size is ~2 MB, well within GitHub and Streamlit Cloud limits.

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

---

## ⚠️ Limitations & Responsible AI

- **Educational prototype only** — not for real lending decisions
- **Small dataset** — only 1,000 historical records from 1994
- **No fairness evaluation** — model may reflect historical biases
- **No model calibration** — probabilities may not be perfectly calibrated
- **Fixed 50% threshold** — not optimized for business objectives
- **Single train/test split** — no cross-validation
- **Feature importance ≠ causation** — SHAP values show association, not causation

See the **Fairness & Responsible AI** page in the application for detailed information.

---

## 📄 License

This project is for educational purposes. The UCI German Credit dataset is publicly available from the UCI Machine Learning Repository.

---

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — German Credit dataset
- [SHAP](https://github.com/shap/shap) — SHapley Additive exPlanations
- [LIME](https://github.com/marcotcr/lime) — Local Interpretable Model-agnostic Explanations
- [XGBoost](https://github.com/dmlc/xgboost) — Gradient boosting framework
- [Streamlit](https://streamlit.io/) — Web application framework
- [fpdf2](https://github.com/py-pdf/fpdf2) — PDF generation library
