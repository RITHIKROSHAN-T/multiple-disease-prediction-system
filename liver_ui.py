# -------------------------------------------------------------
# liver_ui.py
# -------------------------------------------------------------
# Streamlit UI for Liver Disease Prediction.
#
# Responsibilities:
#   - Load the trained liver model from models/liver_model.pkl
#   - Collect user input for liver-related clinical features
#   - Convert categorical fields (like Gender) to numeric values
#   - Build a DataFrame in same structure as training data
#   - Generate prediction and display result + risk probability
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# -------------------------------------------------------------
# Helper: Load trained liver model (cached)
# -------------------------------------------------------------
@st.cache_resource
def load_liver_model():
    model_path = os.path.join("models", "liver_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Liver model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model


# -------------------------------------------------------------
# Feature columns (must match training order EXACTLY)
# NOTE: The original dataset column is misspelled as 'Total_Protiens'
#       so we MUST use the same spelling here.
# -------------------------------------------------------------
LIVER_FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",              # <-- keep this spelling
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]


def build_liver_input_df(
    age,
    gender,
    total_bilirubin,
    direct_bilirubin,
    alk_phos,
    alt,
    ast,
    total_protiens,  # keep var name normal, map to misspelled key
    albumin,
    ag_ratio,
):
    """
    Build a single-row DataFrame for liver model input.
    Column names must match exactly what the model saw during training.
    """
    data = {
        "Age": age,
        "Gender": gender,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alk_phos,
        "Alamine_Aminotransferase": alt,
        "Aspartate_Aminotransferase": ast,
        "Total_Protiens": total_protiens,   # <-- mapped here
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": ag_ratio,
    }

    df_input = pd.DataFrame([data], columns=LIVER_FEATURE_COLUMNS)
    return df_input


# -------------------------------------------------------------
# Main UI function
# -------------------------------------------------------------
def show_liver_page():
    st.subheader("🟠 Liver Disease Prediction")

    st.write(
        """
        Enter patient details below to predict the likelihood of **Liver Disease**  
        using your trained Machine Learning model.
        """
    )

    # Load model
    try:
        model = load_liver_model()
    except Exception as e:
        st.error(f"Error loading liver model: {e}")
        return

    st.markdown("### 🧍 Patient Demographics")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=110, value=45)

    with col2:
        gender_opt = st.selectbox("Gender", ["Male", "Female"])

    # Map Gender to numeric (LabelEncoder likely did: Female=0, Male=1)
    gender_map = {"Female": 0, "Male": 1}
    gender = gender_map[gender_opt]

    st.markdown("### 🧪 Liver Function Test (LFT) Values")

    col3, col4 = st.columns(2)

    with col3:
        total_bilirubin = st.number_input(
            "Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0, step=0.1
        )
        direct_bilirubin = st.number_input(
            "Direct Bilirubin", min_value=0.0, max_value=20.0, value=0.5, step=0.1
        )
        alk_phos = st.number_input(
            "Alkaline Phosphotase", min_value=0, max_value=2000, value=200, step=10
        )

    with col4:
        alt = st.number_input(
            "Alamine Aminotransferase (ALT)", min_value=0, max_value=2000, value=30, step=5
        )
        ast = st.number_input(
            "Aspartate Aminotransferase (AST)", min_value=0, max_value=2000, value=35, step=5
        )

    st.markdown("### 🧬 Proteins")

    col5, col6 = st.columns(2)

    with col5:
        total_protiens = st.number_input(   # label is human-readable, var name is fine
            "Total Proteins", min_value=0.0, max_value=10.0, value=6.5, step=0.1
        )

    with col6:
        albumin = st.number_input(
            "Albumin", min_value=0.0, max_value=6.0, value=3.0, step=0.1
        )

    ag_ratio = st.number_input(
        "Albumin and Globulin Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1
    )

    # ---------------------------------------------------------
    # Predict button
    # ---------------------------------------------------------
    if st.button("🔍 Predict Liver Disease"):
        input_df = build_liver_input_df(
            age=age,
            gender=gender,
            total_bilirubin=total_bilirubin,
            direct_bilirubin=direct_bilirubin,
            alk_phos=alk_phos,
            alt=alt,
            ast=ast,
            total_protiens=total_protiens,   # passes to function
            albumin=albumin,
            ag_ratio=ag_ratio,
        )

        try:
            prediction = model.predict(input_df)[0]

            # Probability if supported
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]  # probability of disease (label=1)
            else:
                prob = None

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error("⚠️ The model predicts **Liver Disease**.")
            else:
                st.success("✅ The model predicts **No Liver Disease**.")

            if prob is not None:
                st.write(f"**Estimated Disease Risk Probability:** `{prob:.2f}`")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
