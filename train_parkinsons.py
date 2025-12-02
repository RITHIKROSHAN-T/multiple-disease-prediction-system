# -------------------------------------------------------------
# train_parkinsons.py
# -------------------------------------------------------------
# Purpose:
#   - Train ML models for Parkinson’s Disease Prediction
#   - Models:
#       * Logistic Regression
#       * Random Forest
#       * XGBoost
#   - Evaluate each model
#   - Select best based on F1-score
#   - Save to: models/parkinsons_model.pkl
#
# Run using:
#   python src/training/train_parkinsons.py
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib


# -------------------------------------------------------------
# Step 1: Load cleaned parkinsons dataset
# -------------------------------------------------------------
def load_dataset():
    data_path = os.path.join("data", "parkinsons_cleaned.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded Parkinson's dataset: {df.shape}")
    return df


# -------------------------------------------------------------
# Step 2: Split into X and y
# -------------------------------------------------------------
def split_features_target(df):
    if "status" not in df.columns:
        raise KeyError("'status' column not found. It should be the target.")

    X = df.drop(columns=["status"])
    y = df["status"]

    print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


# -------------------------------------------------------------
# Step 3: Train three models
# -------------------------------------------------------------
def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train, y_train)
    models["LogisticRegression"] = log_reg

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    return models


# -------------------------------------------------------------
# Step 4: Evaluate a model
# -------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    print(f"\n[INFO] Evaluating: {name}")

    y_pred = model.predict(X_test)

    # Probability for ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "f1": f1,
        "roc": roc,
    }


# -------------------------------------------------------------
# Step 5: Main pipeline
# -------------------------------------------------------------
def main():
    df = load_dataset()
    X, y = split_features_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate
    best_model = None
    best_score = -1
    best_name = ""

    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model
            best_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    output_path = "models/parkinsons_model.pkl"
    joblib.dump(best_model, output_path)

    print(f"\n[INFO] Best model: {best_name} (F1 = {best_score:.4f})")
    print(f"[INFO] Model saved to: {output_path}")


if __name__ == "__main__":
    main()
