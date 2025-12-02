# -------------------------------------------------------------
# app.py
# -------------------------------------------------------------
# Main Streamlit application entry point.
# Provides a sidebar menu to navigate between:
#   - Home
#   - Kidney Disease Prediction
#   - Liver Disease Prediction
#   - Parkinson's Disease Prediction
#
# Each page's UI is implemented in a separate module under app/.
# -------------------------------------------------------------

import streamlit as st

from kidney_ui import show_kidney_page
from liver_ui import show_liver_page
from parkinsons_ui import show_parkinsons_page


def main():
    # Page configuration
    st.set_page_config(
        page_title="Multiple Disease Prediction",
        page_icon="🩺",
        layout="centered"
    )

    # App title
    st.title("🩺 Multiple Disease Prediction System")
    st.markdown(
        """
        This application provides **ML-based predictions** for:
        - Chronic Kidney Disease (CKD)
        - Liver Disease
        - Parkinson's Disease

        Use the sidebar to select a disease and enter patient details.
        """
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Home",
            "Kidney Disease Prediction",
            "Liver Disease Prediction",
            "Parkinson's Disease Prediction",
        ),
    )

    # Render selected page
    if page == "Home":
        show_home_page()
    elif page == "Kidney Disease Prediction":
        show_kidney_page()
    elif page == "Liver Disease Prediction":
        show_liver_page()
    elif page == "Parkinson's Disease Prediction":
        show_parkinsons_page()


def show_home_page():
    """Simple home/overview section."""
    st.subheader("🏠 Home")
    st.write(
        """
        Welcome to the **Multiple Disease Prediction System**.

        **Available modules:**
        - ✅ Kidney Disease Prediction
        - ✅ Liver Disease Prediction
        - ✅ Parkinson's Disease Prediction
        """
    )
    st.info("Use the sidebar on the left to select a prediction module.")


if __name__ == "__main__":
    main()
