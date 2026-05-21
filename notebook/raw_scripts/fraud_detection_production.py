"""
Fraud Detection - Production Grade Pipeline
============================================
Reusable functions for:
- Data loading & validation
- EDA
- Train/test split
- Pipeline creation (ColumnTransformer + XGBoost)
- Hyperparameter tuning
- Evaluation
- SHAP explainability
- Model saving & loading

Author: Allen Harry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════

def get_engine():
    """Create SQL Server connection engine."""
    from sqlalchemy import create_engine

    engine = create_engine(
        "mssql+pyodbc://@ALLEN-TIDER/CREDT_SCORE?"
        "driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    )
    return engine


def fetch_data(query):
    """
    Fetch data from SQL Server.

    Args:
        query (str): SQL query

    Returns:
        pd.DataFrame or None
    """
    try:
        engine = get_engine()
        df = pd.read_sql(query, engine)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SECTION 2: DATA VALIDATION & QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════

def validate_data(df, target_col="Class"):
    """
    Run data quality checks. Call this immediately after loading.

    Args:
        df (pd.DataFrame): Raw dataframe
        target_col (str): Target column name

    Returns:
        dict: Quality report
    """
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df[target_col].value_counts(normalize=True).to_dict(),
    }

    print(f"Shape: {report['shape']}")
    print(f"Missing values: {report['total_missing']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 3: DATA CLEANING (Safe before split — no stats learned)
# ═══════════════════════════════════════════════════════════════

def clean_data(df, drop_cols=None):
    """
    Clean data: drop columns, remove duplicates, fix types.
    These operations are safe BEFORE splitting (no statistics learned).

    Args:
        df (pd.DataFrame): Input dataframe
        drop_cols (list): Columns to drop

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    initial_shape = df.shape

    # Drop specified columns
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
        print(f"Dropped columns: {existing}")

    # Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count} duplicate rows")

    # Ensure Amount is numeric
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    print(f"Shape: {initial_shape} → {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 4: TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════

def split_data(df, target_col="Class", test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    Uses stratification to preserve class balance.

    Args:
        df (pd.DataFrame): Cleaned dataframe
        target_col (str): Target column name
        test_size (float): Proportion for test set
        random_state (int): Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"y_test distribution:  {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
# SECTION 5: PIPELINE CREATION & TRAINING
# ═══════════════════════════════════════════════════════════════

def create_pipeline(y_train, amount_column="Amount"):
    """
    Create production pipeline: ColumnTransformer + XGBoost.
    Scaling happens INSIDE the pipeline (no leakage).

    Args:
        y_train (pd.Series): Training target (for class weight calculation)
        amount_column (str): Column to scale

    Returns:
        Pipeline: Untrained pipeline
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier

    # Calculate class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Preprocessor: scale Amount, pass through everything else
    # Use set_output to force numpy array output (avoids dtype issues with SHAP)
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", RobustScaler(), [amount_column])
        ],
        remainder="passthrough"
    ).set_output(transform="default")

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    return pipeline


def train_pipeline(pipeline, X_train, y_train):
    """
    Train the pipeline.

    Args:
        pipeline: sklearn Pipeline
        X_train: Training features
        y_train: Training target

    Returns:
        Pipeline: Trained pipeline
    """
    pipeline.fit(X_train, y_train)
    print("Pipeline trained successfully.")
    return pipeline


# ═══════════════════════════════════════════════════════════════
# SECTION 6: HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════

def tune_pipeline(X_train, y_train, amount_column="Amount", n_iter=50, cv=5):
    """
    Tune pipeline hyperparameters using RandomizedSearchCV.

    Uses refit=True (default), so best_estimator_ is already trained
    on ALL of X_train with the best params. No manual retrain needed.

    Args:
        X_train: Training features
        y_train: Training target
        amount_column (str): Column to scale
        n_iter (int): Number of random combinations to try
        cv (int): Number of cross-validation folds

    Returns:
        Pipeline: Best trained pipeline (already fitted)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from xgboost import XGBClassifier

    # Class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", RobustScaler(), [amount_column])
        ],
        remainder="passthrough"
    ).set_output(transform="default")

    # Pipeline with placeholder params (will be overridden by search)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight
        ))
    ])

    # Parameter grid (model__ prefix because model is inside pipeline)
    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
    }

    # RandomizedSearchCV with refit=True (auto-retrains best model on full data)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="average_precision",  # PR-AUC — best for imbalanced data
        cv=cv_strategy,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True  # Auto-retrains best model on ALL X_train
    )

    search.fit(X_train, y_train)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")

    # best_estimator_ is already trained on full X_train (refit=True)
    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════
# SECTION 7: PREDICTION & THRESHOLD
# ═══════════════════════════════════════════════════════════════

def get_predictions(pipeline, X_test, threshold=0.15):
    """
    Get predictions from trained pipeline.

    Args:
        pipeline: Trained pipeline
        X_test: Test features
        threshold (float): Decision threshold

    Returns:
        tuple: (y_probs, y_pred)
            - y_probs: probability of fraud (0.0 to 1.0)
            - y_pred: binary prediction (0 or 1)
    """
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"Threshold: {threshold}")
    print(f"Predicted fraud: {y_pred.sum()} / {len(y_pred)} ({y_pred.mean()*100:.2f}%)")

    return y_probs, y_pred


def find_optimal_threshold(y_test, y_probs, target_recall=0.85):
    """
    Find threshold that achieves target recall.

    Args:
        y_test: True labels
        y_probs: Predicted probabilities
        target_recall (float): Desired recall level

    Returns:
        float: Optimal threshold
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    # Find threshold closest to target recall
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    optimal_threshold = thresholds[idx]

    print(f"Target recall: {target_recall}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Precision at this threshold: {precisions[idx]:.4f}")
    print(f"Actual recall at this threshold: {recalls[idx]:.4f}")

    return optimal_threshold


# ═══════════════════════════════════════════════════════════════
# SECTION 8: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(y_test, y_pred, y_probs):
    """
    Complete model evaluation with all relevant metrics.

    Args:
        y_test: True labels
        y_pred: Binary predictions (uses threshold)
        y_probs: Probability predictions

    Returns:
        dict: All metrics
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score
    )

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_probs),
        "pr_auc": average_precision_score(y_test, y_probs),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def plot_precision_recall_curve(y_test, y_probs):
    """Plot precision-recall curve with threshold markers."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PR Curve
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall Curve (AUC={pr_auc:.4f})")
    axes[0].grid(True, alpha=0.3)

    # Precision & Recall vs Threshold
    axes[1].plot(thresholds, precisions[:-1], 'b-', label="Precision")
    axes[1].plot(thresholds, recalls[:-1], 'r-', label="Recall")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision & Recall vs Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# SECTION 9: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def get_feature_importance(pipeline, X_train, top_n=15):
    """
    Extract and plot feature importance from trained pipeline.

    Args:
        pipeline: Trained pipeline
        X_train: Training features (for column names)
        top_n (int): Number of top features to show

    Returns:
        pd.DataFrame: Feature importance table
    """
    model = pipeline.named_steps["model"]

    # Get feature names after ColumnTransformer
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in feature_names]

    importance_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"].head(top_n)[::-1],
        importance_df["Importance"].head(top_n)[::-1]
    )
    plt.title(f"Top {top_n} Feature Importance (XGBoost Gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} Features:")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


# ═══════════════════════════════════════════════════════════════
# SECTION 10: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

def explain_with_shap(pipeline, X_test, n_samples=200):
    """
    Generate SHAP explanations for the model.

    This function completely bypasses the ColumnTransformer to avoid
    sklearn dtype issues. It manually scales Amount using the same
    median/IQR the pipeline learned during training.

    Args:
        pipeline: Trained pipeline
        X_test: Test features (raw DataFrame)
        n_samples (int): Number of samples to explain

    Returns:
        tuple: (shap_values, X_display_df)
    """
    import shap

    # Step 1: Extract the XGBoost model
    model = pipeline.named_steps["model"]

    # Step 2: Get the scaler parameters from inside the pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    # After fitting, ColumnTransformer stores fitted transformers in transformers_
    fitted_scaler = preprocessor.transformers_[0][1]  # The RobustScaler
    scaler_center = fitted_scaler.center_[0]  # median
    scaler_scale = fitted_scaler.scale_[0]    # IQR

    # Step 3: Manually scale Amount (simple math — no sklearn call needed)
    X_sample = X_test.iloc[:n_samples].copy().reset_index(drop=True)

    # Apply RobustScaler formula: (x - median) / IQR
    amount_scaled = (X_sample["Amount"].values.astype(float) - scaler_center) / scaler_scale

    # Step 4: Build the numpy array in the same column order as ColumnTransformer output
    # ColumnTransformer output order: [Amount_scaled, then all other columns in original order]
    other_cols = [c for c in X_sample.columns if c != "Amount"]
    other_values = X_sample[other_cols].values.astype(float)

    # Combine: Amount first, then the rest
    X_final = np.column_stack([amount_scaled, other_values])
    feature_names = ["Amount"] + other_cols

    # Step 5: Verify shape matches what model expects
    print(f"Input shape for SHAP: {X_final.shape}")
    print(f"Model expects: {model.n_features_in_} features")

    # Step 6: SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    print(f"SHAP values shape: {shap_values.shape}")

    # Step 7: Create display DataFrame for plots
    X_display = pd.DataFrame(X_final, columns=feature_names)

    # Step 8: Plots
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_display, show=True)

    return shap_values, X_display


def explain_single_prediction(pipeline, X_test, sample_idx=0):
    """
    Explain why the model made a specific prediction.

    Args:
        pipeline: Trained pipeline
        X_test: Test features (raw DataFrame)
        sample_idx (int): Which row to explain

    Returns:
        pd.DataFrame: Feature contributions for this prediction
    """
    import shap

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    fitted_scaler = preprocessor.transformers_[0][1]
    scaler_center = fitted_scaler.center_[0]
    scaler_scale = fitted_scaler.scale_[0]

    # Prepare single sample
    row = X_test.iloc[sample_idx]
    amount_scaled = (float(row["Amount"]) - scaler_center) / scaler_scale

    other_cols = [c for c in X_test.columns if c != "Amount"]
    other_values = row[other_cols].values.astype(float)

    X_single = np.concatenate([[amount_scaled], other_values]).reshape(1, -1)
    feature_names = ["Amount"] + other_cols

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    # Explanation table
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Value": X_single[0],
        "SHAP_Value": shap_values[0],
        "Direction": ["Increases fraud risk" if s > 0 else "Decreases fraud risk"
                      for s in shap_values[0]]
    }).sort_values("SHAP_Value", key=abs, ascending=False)

    # Print
    prob = pipeline.predict_proba(X_test.iloc[[sample_idx]])[:, 1][0]
    print(f"Prediction: {prob:.4f} probability of fraud")
    print(f"\nTop 5 reasons:")
    print(explanation.head(5).to_string(index=False))

    # Waterfall plot
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_single[0],
        feature_names=feature_names
    ))

    return explanation


# ═══════════════════════════════════════════════════════════════
# SECTION 11: SAVE & LOAD MODEL
# ═══════════════════════════════════════════════════════════════

def save_model(pipeline, metrics, threshold, output_dir="artifacts"):
    """
    Save trained pipeline and metadata for production deployment.

    Args:
        pipeline: Trained pipeline
        metrics (dict): Evaluation metrics
        threshold (float): Decision threshold
        output_dir (str): Output directory

    Returns:
        str: Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save pipeline (one object = preprocessor + model)
    model_path = os.path.join(output_dir, "fraud_pipeline.pkl")
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_version": timestamp,
        "model_type": "Pipeline(ColumnTransformer + XGBClassifier)",
        "threshold": threshold,
        "metrics": metrics,
        "training_date": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    return model_path


def load_model(model_path="artifacts/fraud_pipeline.pkl"):
    """
    Load trained pipeline for inference.

    Args:
        model_path (str): Path to saved pipeline

    Returns:
        Pipeline: Trained pipeline ready for predictions
    """
    pipeline = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return pipeline


# ═══════════════════════════════════════════════════════════════
# SECTION 12: PRODUCTION INFERENCE
# ═══════════════════════════════════════════════════════════════

def predict_fraud(pipeline, input_data, threshold=0.15):
    """
    Production inference function.
    Pass raw data — pipeline handles all preprocessing.

    Args:
        pipeline: Trained pipeline
        input_data (pd.DataFrame): Raw features (same columns as training)
        threshold (float): Decision threshold

    Returns:
        pd.DataFrame: Predictions with probabilities
    """
    y_probs = pipeline.predict_proba(input_data)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    results = pd.DataFrame({
        "probability": y_probs,
        "prediction": y_pred,
        "risk_level": pd.cut(
            y_probs,
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High", "Critical"]
        )
    })

    return results
