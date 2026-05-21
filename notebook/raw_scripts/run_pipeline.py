"""
Fraud Detection - Run Pipeline
================================
This script shows how to use the production functions.
Run this in your notebook cell by cell, or as a script.

Usage in notebook:
    %run fraud_detection_production.py
    Then run the cells below.
"""

# ═══════════════════════════════════════════════════════════════
# STEP 1: IMPORTS & LOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════
from notebook.raw_scripts.fraud_detection_production import *

# ═══════════════════════════════════════════════════════════════
# STEP 2: LOAD DATA
# ═══════════════════════════════════════════════════════════════
query = "SELECT * FROM dbo.creditcard_data"
df = fetch_data(query)

# ═══════════════════════════════════════════════════════════════
# STEP 3: VALIDATE DATA
# ═══════════════════════════════════════════════════════════════
report = validate_data(df)

# ═══════════════════════════════════════════════════════════════
# STEP 4: CLEAN DATA (safe before split — no stats learned)
# ═══════════════════════════════════════════════════════════════
df = clean_data(df, drop_cols=["Time"])

# ═══════════════════════════════════════════════════════════════
# STEP 5: SPLIT DATA (BEFORE any preprocessing that learns stats)
# ═══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = split_data(df)

# ═══════════════════════════════════════════════════════════════
# STEP 6A: QUICK BASELINE (default params, no tuning)
# ═══════════════════════════════════════════════════════════════
print("\n--- BASELINE MODEL ---")
baseline_pipeline = create_pipeline(y_train)
baseline_pipeline = train_pipeline(baseline_pipeline, X_train, y_train)

y_probs_baseline, y_pred_baseline = get_predictions(baseline_pipeline, X_test, threshold=0.15)
baseline_metrics = evaluate_model(y_test, y_pred_baseline, y_probs_baseline)

# ═══════════════════════════════════════════════════════════════
# STEP 6B: TUNED MODEL (hyperparameter search)
# ═══════════════════════════════════════════════════════════════
print("\n--- TUNING MODEL ---")
best_pipeline = tune_pipeline(X_train, y_train, n_iter=50, cv=5)

# ═══════════════════════════════════════════════════════════════
# STEP 7: FIND OPTIMAL THRESHOLD
# ═══════════════════════════════════════════════════════════════
y_probs = best_pipeline.predict_proba(X_test)[:, 1]
optimal_threshold = find_optimal_threshold(y_test, y_probs, target_recall=0.85)

# ═══════════════════════════════════════════════════════════════
# STEP 8: FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════
y_probs, y_pred = get_predictions(best_pipeline, X_test, threshold=optimal_threshold)
metrics = evaluate_model(y_test, y_pred, y_probs)

# Plot PR curve
plot_precision_recall_curve(y_test, y_probs)

# ═══════════════════════════════════════════════════════════════
# STEP 9: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
importance_df = get_feature_importance(best_pipeline, X_train)

# ═══════════════════════════════════════════════════════════════
# STEP 10: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
shap_values, X_shap = explain_with_shap(best_pipeline, X_test, n_samples=200)

# Explain a single prediction
explanation = explain_single_prediction(best_pipeline, X_test, sample_idx=0)

# ═══════════════════════════════════════════════════════════════
# STEP 11: SAVE MODEL FOR PRODUCTION
# ═══════════════════════════════════════════════════════════════
save_model(best_pipeline, metrics, threshold=optimal_threshold)

# ═══════════════════════════════════════════════════════════════
# STEP 12: PRODUCTION INFERENCE (how to use the saved model)
# ═══════════════════════════════════════════════════════════════
# Load model
loaded_pipeline = load_model("artifacts/fraud_pipeline.pkl")

# Predict on new data (pass raw data — pipeline handles preprocessing)
sample_data = X_test.iloc[:5]
results = predict_fraud(loaded_pipeline, sample_data, threshold=optimal_threshold)
print("\nProduction Predictions:")
print(results)
