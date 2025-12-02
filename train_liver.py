# -------------------------------------------------------------
# train_liver.py
# -------------------------------------------------------------
# Purpose:
#   - Train ML models for Liver Disease Prediction
#   - Algorithms:
#       * Logistic Regression
#       * Random Forest
#       * XGBoost
#   - Select best model based on F1-score
#   - Save best model as: models/liver_model.pkl
#
# Input:
#   - data/liver_cleaned.csv   (from 02_eda_liver.ipynb)
#
# Run from project root:
#   python src/training/train_liver.py
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
# Step 1: Load cleaned liver dataset
# -------------------------------------------------------------
def load_liver_data():
    """
    Load the cleaned liver dataset from the data/ folder.
    """
    data_path = os.path.join("data", "liver_cleaned.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned liver dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded liver dataset from {data_path}")
    print(f"[INFO] Shape: {df.shape}")
    return df


# -------------------------------------------------------------
# Step 2: Split into features (X) and target (y)
# -------------------------------------------------------------
def split_features_target(df: pd.DataFrame):
    """
    Separate features (X) and target (y).

    Assumptions for Indian Liver Patient dataset:
      - Target column is 'Dataset'
        * Usually: 1 = liver disease, 2 = no disease
      - We convert it to binary 0/1:
        * 1 -> 1 (disease)
        * 2 -> 0 (no disease)
    """
    cols = df.columns.tolist()

    if "Dataset" not in cols:
        raise KeyError("'Dataset' (target) column not found in liver dataset.")

    # Features: all except target
    X = df.drop(columns=["Dataset"])
    y = df["Dataset"].copy()

    # Map labels 1,2 -> 1,0 if needed
    if y.nunique() > 2 or sorted(y.unique()) not in ([0, 1], [1, 2]):
        print("[WARN] Unexpected target values in 'Dataset'. Please inspect manually.")
    else:
        # Typical mapping for Indian Liver Patient dataset
        if sorted(y.unique()) == [1, 2]:
            y = y.map({1: 1, 2: 0})

    print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"[INFO] Target value counts:\n{y.value_counts()}")
    return X, y


# -------------------------------------------------------------
# Step 3: Train multiple models
# -------------------------------------------------------------
def train_models(X_train, y_train):
    """
    Train:
      - Logistic Regression
      - Random Forest
      - XGBoost

    Returns:
      dict: {model_name: model_object}
    """
    models = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    models["LogisticRegression"] = log_reg

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    return models


# -------------------------------------------------------------
# Step 4: Evaluate a single model
# -------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    """
    Compute metrics for a given model.
    """
    print(f"\n[INFO] Evaluating model: {name}")

    y_pred = model.predict(X_test)

    # ROC-AUC if probabilities available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
    }


# -------------------------------------------------------------
# Step 5: Main training pipeline
# -------------------------------------------------------------
def main():
    # 1️⃣ Load data
    df = load_liver_data()

    # 2️⃣ Split into X, y
    X, y = split_features_target(df)

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4️⃣ Train models
    models = train_models(X_train, y_train)

    # 5️⃣ Evaluate & select best model by F1-score
    best_model_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
            best_model = model

    # 6️⃣ Save best model
    if best_model is not None:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "liver_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"\n[INFO] Best liver model: {best_model_name} (F1 = {best_f1:.4f})")
        print(f"[INFO] Saved best model to: {model_path}")
    else:
        print("[ERROR] No best model found. Please check training process.")


if __name__ == "__main__":
    main()
