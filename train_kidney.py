# -------------------------------------------------------------
# train_kidney.py  (RAW DATA VERSION - MATCHES STREAMLIT INPUT)
# -------------------------------------------------------------
# Purpose:
#   - Load the original kidney_disease.csv (raw values)
#   - Clean + encode data INSIDE this script
#   - NO SCALING (so Streamlit inputs like age=45, bp=80 match)
#   - Train Logistic Regression, Random Forest, XGBoost
#   - Save the best model as models/kidney_model.pkl
#
# Run from project root:
#   python src/training/train_kidney.py
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib

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

DATA_PATH = os.path.join("data", "kidney_disease - kidney_disease (1).csv")  # <-- adjust name if different


def load_raw_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Raw kidney dataset not found at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded raw kidney dataset: {df.shape}")
    return df


def preprocess_kidney(df: pd.DataFrame):
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Drop rows where target is missing
    df = df.dropna(subset=["classification"])

    # Handle weird 'ckd\\t' etc
    df["classification"] = df["classification"].replace({"ckd\t": "ckd"})

    # Categorical mappings - MUST match Streamlit UI
    binary_maps = {
        "rbc": {"abnormal": 0, "normal": 1},
        "pc": {"abnormal": 0, "normal": 1},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"no": 0, "yes": 1},
        "dm": {"no": 0, "yes": 1},
        "cad": {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe": {"no": 0, "yes": 1},
        "ane": {"no": 0, "yes": 1},
        "classification": {"notckd": 0, "ckd": 1, "no": 0},
    }

    # Fill missing values first (mode for object, median for numeric)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Apply manual binary mappings where possible
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # For any remaining object columns, factorize (just in case)
    remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
    for col in remaining_obj:
        df[col], _ = pd.factorize(df[col])

    # Define X and y
    if "classification" not in df.columns:
        raise KeyError("'classification' column missing after preprocessing")

    drop_cols = ["classification"]
    if "id" in df.columns:
        drop_cols.append("id")

    X = df.drop(columns=drop_cols)
    y = df["classification"].astype(int)

    print(f"[INFO] After preprocessing: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def train_models(X_train, y_train):
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


def evaluate_model(name, model, X_test, y_test):
    print(f"\n[INFO] Evaluating model: {name}")
    y_pred = model.predict(X_test)

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

    return f1


def main():
    # 1) Load raw data
    df_raw = load_raw_data()

    # 2) Preprocess to numeric X, y (NO SCALING)
    X, y = preprocess_kidney(df_raw)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4) Train models
    models = train_models(X_train, y_train)

    # 5) Evaluate and pick best by F1
    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        f1 = evaluate_model(name, model, X_test, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # 6) Save best model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "kidney_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"\n[INFO] Best kidney model: {best_name} (F1 = {best_f1:.4f})")
    print(f"[INFO] Saved kidney model to: {model_path}")


if __name__ == "__main__":
    main()
