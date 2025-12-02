# -------------------------------------------------------------
# kidney_ui.py
# -------------------------------------------------------------
# Streamlit UI for Kidney Disease Prediction.
#
# Responsibilities:
#   - Load the trained kidney model from models/kidney_model.pkl
#   - Collect user input for kidney-related features
#   - Convert categorical inputs to numeric (0/1) as used in training
#   - Build a single-row DataFrame in the correct column order
#   - Pass the data to the model and display the prediction result
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# -------------------------------------------------------------
# Helper: Load trained kidney model (cached)
# -------------------------------------------------------------
@st.cache_resource
def load_kidney_model():
    model_path = os.path.join("models", "kidney_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kidney model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model


# -------------------------------------------------------------
# Helper: Prepare input as DataFrame in correct order
# Feature list must match the training X columns:
#   (kidney_disease dataset without 'id' and 'classification')
# -------------------------------------------------------------
KIDNEY_FEATURE_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]


def build_input_dataframe(
    age,
    bp,
    sg,
    al,
    su,
    rbc,
    pc,
    pcc,
    ba,
    bgr,
    bu,
    sc,
    sod,
    pot,
    hemo,
    pcv,
    wc,
    rc,
    htn,
    dm,
    cad,
    appet,
    pe,
    ane,
):
    """
    Create a single-row DataFrame in the same format as the model training data.
    """
    data = {
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wc": wc,
        "rc": rc,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane,
    }

    df_input = pd.DataFrame([data], columns=KIDNEY_FEATURE_COLUMNS)
    return df_input


# -------------------------------------------------------------
# Main UI function
# -------------------------------------------------------------
def show_kidney_page():
    st.subheader("🟢 Kidney Disease Prediction (CKD)")

    st.write(
        """
        Enter patient details below to predict the likelihood of **Chronic Kidney Disease (CKD)**.
        The prediction is based on your trained Machine Learning model.
        """
    )

    # Load model
    try:
        model = load_kidney_model()
    except Exception as e:
        st.error(f"Error loading kidney model: {e}")
        return

    st.markdown("### 🧍 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=110, value=45)
        bp = st.number_input("Blood Pressure (bp)", min_value=0, max_value=200, value=80)
        sg = st.number_input("Specific Gravity (sg)", min_value=1.00, max_value=1.05, value=1.02, step=0.01, format="%.2f")
        al = st.number_input("Albumin (al)", min_value=0, max_value=5, value=1)
        su = st.number_input("Sugar (su)", min_value=0, max_value=5, value=0)

    with col2:
        bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0, max_value=500, value=120)
        bu = st.number_input("Blood Urea (bu)", min_value=0, max_value=300, value=54)
        sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=25.0, value=1.2, step=0.1)
        sod = st.number_input("Sodium (sod)", min_value=0.0, max_value=200.0, value=135.0, step=0.5)
        pot = st.number_input("Potassium (pot)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)

    st.markdown("### 🩸 Lab / Clinical Indicators")

    col3, col4 = st.columns(2)

    with col3:
        hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume (pcv)", min_value=0, max_value=70, value=40)
        wc = st.number_input("White Blood Cell Count (wc)", min_value=0, max_value=30000, value=8000)
        rc = st.number_input("Red Blood Cell Count (rc)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)

    with col4:
        rbc_opt = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
        pc_opt = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
        pcc_opt = st.selectbox("Pus Cell Clumps (pcc)", ["notpresent", "present"])
        ba_opt = st.selectbox("Bacteria (ba)", ["notpresent", "present"])

    st.markdown("### 🧬 Medical History")

    col5, col6 = st.columns(2)

    with col5:
        htn_opt = st.selectbox("Hypertension (htn)", ["no", "yes"])
        dm_opt = st.selectbox("Diabetes Mellitus (dm)", ["no", "yes"])
        cad_opt = st.selectbox("Coronary Artery Disease (cad)", ["no", "yes"])

    with col6:
        appet_opt = st.selectbox("Appetite (appet)", ["good", "poor"])
        pe_opt = st.selectbox("Pedal Edema (pe)", ["no", "yes"])
        ane_opt = st.selectbox("Anemia (ane)", ["no", "yes"])

    # ---------------------------------------------------------
    # Map categorical inputs to numeric values
    # (Must match LabelEncoder behavior used during training)
    # ---------------------------------------------------------
    rbc_map = {"abnormal": 0, "normal": 1}
    pc_map = {"abnormal": 0, "normal": 1}
    pcc_map = {"notpresent": 0, "present": 1}
    ba_map = {"notpresent": 0, "present": 1}
    yn_map = {"no": 0, "yes": 1}
    appet_map = {"good": 0, "poor": 1}

    rbc = rbc_map[rbc_opt]
    pc = pc_map[pc_opt]
    pcc = pcc_map[pcc_opt]
    ba = ba_map[ba_opt]
    htn = yn_map[htn_opt]
    dm = yn_map[dm_opt]
    cad = yn_map[cad_opt]
    appet = appet_map[appet_opt]
    pe = yn_map[pe_opt]
    ane = yn_map[ane_opt]

    # ---------------------------------------------------------
    # Predict button
    # ---------------------------------------------------------
    if st.button("🔍 Predict Kidney Disease"):
        # Build model input
        input_df = build_input_dataframe(
            age=age,
            bp=bp,
            sg=sg,
            al=al,
            su=su,
            rbc=rbc,
            pc=pc,
            pcc=pcc,
            ba=ba,
            bgr=bgr,
            bu=bu,
            sc=sc,
            sod=sod,
            pot=pot,
            hemo=hemo,
            pcv=pcv,
            wc=wc,
            rc=rc,
            htn=htn,
            dm=dm,
            cad=cad,
            appet=appet,
            pe=pe,
            ane=ane,
        )

        try:
            prediction = model.predict(input_df)[0]

            # Try to get probability if available
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]  # probability of CKD
            else:
                prob = None

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error("⚠️ The model predicts **Chronic Kidney Disease (CKD)**.")
            else:
                st.success("✅ The model predicts **No Chronic Kidney Disease (No CKD)**.")

            if prob is not None:
                st.write(f"**Estimated CKD Risk Probability:** `{prob:.2f}`")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
