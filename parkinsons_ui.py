# -------------------------------------------------------------
# parkinsons_ui.py
# -------------------------------------------------------------
# Streamlit UI for Parkinson's Disease Prediction.
#
# Responsibilities:
#   - Load trained model from models/parkinsons_model.pkl
#   - Take 22 voice-measure inputs
#   - Build DataFrame in exact feature order
#   - Output prediction + probability
# -------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------------------
# Load the parkinsons model (cached)
# -------------------------------------------------------------
@st.cache_resource
def load_parkinsons_model():
    model_path = os.path.join("models", "parkinsons_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Parkinson model file not found.")
    return joblib.load(model_path)


# -------------------------------------------------------------
# Feature names in EXACT order used during training
# -------------------------------------------------------------
PARKINSON_FEATURES = [
    'MDVP:Fo(Hz)',
    'MDVP:Fhi(Hz)',
    'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)',
    'MDVP:RAP',
    'MDVP:PPQ',
    'Jitter:DDP',
    'MDVP:Shimmer',
    'MDVP:Shimmer(dB)',
    'Shimmer:APQ3',
    'Shimmer:APQ5',
    'MDVP:APQ',
    'Shimmer:DDA',
    'NHR',
    'HNR',
    'RPDE',
    'DFA',
    'spread1',
    'spread2',
    'D2',
    'PPE'
]


def build_input_df(values):
    """
    Build a DataFrame with a single row in correct feature order.
    """
    df = pd.DataFrame([values], columns=PARKINSON_FEATURES)
    return df


# -------------------------------------------------------------
# UI function
# -------------------------------------------------------------
def show_parkinsons_page():
    st.subheader("🟣 Parkinson's Disease Prediction")

    st.write("""
    Enter the voice measurement values below.  
    These features are extracted from vocal frequency and amplitude variations.
    """)

    model = load_parkinsons_model()

    inputs = {}
    cols = st.columns(2)

    # Build inputs in two-column layout
    for i, feature in enumerate(PARKINSON_FEATURES):
        col = cols[i % 2]
        inputs[feature] = col.number_input(
            feature,
            value=0.0,
            format="%.5f"
        )

    if st.button("🔍 Predict Parkinson's Disease"):
        df_input = build_input_df(list(inputs.values()))

        prediction = model.predict(df_input)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_input)[0][1]
        else:
            prob = None

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ The model predicts **Parkinson's Disease**.")
        else:
            st.success("✅ The model predicts **No Parkinson's Disease**.")

        if prob is not None:
            st.write(f"**Probability of Disease:** `{prob:.2f}`")
