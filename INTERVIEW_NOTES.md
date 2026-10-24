# CreditLens AI — Interview Notes

## 30-Second Explanation

CreditLens AI is an educational machine-learning project that predicts whether a credit applicant is likely to default or get approved. It uses XGBoost and MLP models trained on the UCI German Credit dataset, and explains every prediction using SHAP and LIME — so you can see *why* the model made its decision, not just *what* it predicted.

---

## 1-Minute Explanation

CreditLens AI addresses the "black box" problem in credit risk assessment. When a bank uses a machine learning model to evaluate loan applications, the applicant and the bank officer both need to understand *why* a decision was made.

The project uses two models — XGBoost (gradient boosting) and an MLP neural network — trained on the UCI German Credit dataset with 1,000 records and 20 features. After preprocessing with StandardScaler and OneHotEncoder, the data has 61 transformed features.

For every prediction, SHAP shows how each feature pushed the result toward approval or default, and LIME creates a local approximation around that specific decision. The dashboard also includes what-if analysis, stress testing, model comparison, and PDF report generation.

---

## 2-Minute Explanation

CreditLens AI is an end-to-end explainable credit risk prototype built as a B.Tech mini-project.

**Problem:** Credit risk models are often black boxes. Applicants and regulators need to understand why a loan is approved or denied.

**Dataset:** UCI Statlog German Credit — 1,000 records, 7 numeric features (age, credit amount, duration, etc.), 13 categorical features (checking account status, credit history, loan purpose, etc.). Target: good credit (1) or bad credit (0).

**Preprocessing:** A scikit-learn ColumnTransformer applies StandardScaler to numeric features and OneHotEncoder (with handle_unknown='ignore') to categorical features. Fitted only on training data. Results in 61 transformed features.

**Models:**
- XGBoost (primary): 150 trees, max depth 4, learning rate 0.05. ROC-AUC: 0.761
- MLP (comparison): 64→32 hidden layers, ReLU, Adam, early stopping. ROC-AUC: 0.709

**Explainability:**
- SHAP (TreeExplainer): Shows feature-level contributions to each prediction
- LIME (1,200 perturbations): Creates local linear approximation

**Analysis Tools:** What-if scenarios (vary loan amount/duration), stress testing (one-variable sensitivity), and global feature importance.

**Limitations:** Small historical dataset, no cross-validation, no fairness evaluation, fixed 50% threshold. This is an educational prototype, not a production system.

---

## Common Interview Questions & Answers

### Q: What is the problem statement?
Credit risk assessment needs to be both accurate *and* explainable. CreditLens AI demonstrates how ML models can predict credit risk while providing transparent explanations for each decision.

### Q: Why this dataset?
The UCI German Credit dataset is a well-known benchmark for credit scoring. It has 1,000 records with realistic credit features. It's small enough for a mini-project but realistic enough to demonstrate ML concepts.

### Q: Why XGBoost?
XGBoost is one of the best-performing algorithms for structured/tabular data. It handles feature interactions well, is robust to overfitting (with proper max_depth), and is compatible with SHAP TreeExplainer for fast, exact explanations.

### Q: Why MLP?
To provide a neural network comparison point. MLP represents a fundamentally different approach (neural network vs. ensemble of trees) and shows different precision-recall trade-offs.

### Q: Why both models?
To demonstrate that different algorithms make different trade-offs. XGBoost has higher precision (77.5%), while MLP has much higher recall (98.6%). Neither is universally "best."

### Q: Why StandardScaler?
Neural networks (MLP) and many algorithms perform better when numeric features are on similar scales. StandardScaler normalizes to mean=0, std=1.

### Q: Why OneHotEncoder?
Categorical features like "checking account status" need to be converted to numeric form. OneHotEncoder creates binary columns for each category. We use handle_unknown='ignore' so the model can handle unseen categories at prediction time.

### Q: What is SHAP?
SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It assigns each feature an importance value for a particular prediction. The key property is that SHAP values sum to the difference between the prediction and the baseline (average prediction).

### Q: What is LIME?
LIME (Local Interpretable Model-agnostic Explanations) creates a simple, interpretable model around a specific prediction by perturbing the input and fitting a linear model. It works with any model (model-agnostic).

### Q: SHAP vs LIME?
- SHAP: Theoretically grounded (Shapley values), gives exact values for tree models, global and local explanations
- LIME: Model-agnostic, creates local approximations, may give different results than SHAP
- Both are complementary — they confirm or challenge each other's explanations

### Q: What is ROC-AUC?
ROC-AUC measures the model's ability to distinguish between classes across all thresholds. AUC=1.0 is perfect, AUC=0.5 is random. Our XGBoost gets 0.761, meaning it correctly ranks a random good-credit applicant higher than a random bad-credit applicant about 76% of the time.

### Q: What does 60% approval probability mean?
It means the model estimates a 60% chance that this specific applicant has good credit. This is about THIS applicant — it's NOT the model's overall accuracy.

### Q: What does 76% ROC-AUC mean?
It describes the model's OVERALL discriminative performance across ALL test samples. It is NOT the probability for any specific applicant.

### Q: What is the 50% threshold?
If the approval probability ≥ 50%, we classify as "likely approval." This is a fixed project threshold, not optimized or industry-standard. In practice, this threshold would be tuned based on the cost of false positives vs. false negatives.

### Q: What is the what-if analysis?
An illustrative bounded sensitivity analysis. We change one variable (e.g., credit amount) and see how the prediction changes. It shows the model's sensitivity, NOT a guarantee of outcomes. It is NOT financial advice.

### Q: What is stress testing?
One-variable-at-a-time sensitivity analysis. We sweep a variable across its range while holding everything else constant. This shows how sensitive the model is to individual features.

### Q: What are the limitations?
- Small dataset (1,000 records from 1994)
- Single 80/20 split, no cross-validation
- No hyperparameter optimization
- Fixed 50% threshold
- No fairness evaluation
- SHAP only for XGBoost
- Educational prototype, not production-ready

### Q: Why is this not production-ready?
- No regulatory compliance (FCRA, ECOA)
- No formal bias testing
- No model monitoring or drift detection
- No calibration
- No real-time data pipeline
- Historical dataset may not reflect current patterns
- Would need human oversight for all decisions

---

## Key Technical Details to Remember

| Aspect | Detail |
|--------|--------|
| Dataset | UCI German Credit, 1000 records, 20 features |
| Split | 80/20, stratified, random_state=42 |
| Preprocessing | StandardScaler + OneHotEncoder → 61 features |
| XGBoost | 150 trees, depth 4, lr 0.05, ROC-AUC ≈ 0.761 |
| MLP | 64→32, ReLU, Adam, early stopping, ROC-AUC ≈ 0.709 |
| SHAP | TreeExplainer, tree_path_dependent |
| LIME | 1,200 perturbations, 10 features |
| Threshold | 50% fixed |
| Tests | 55 unit tests, all passing |
