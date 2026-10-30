
from pathlib import Path

import streamlit as st
from src.predict import load_model, predict_complaint


MODEL_PATH = Path("models/ticket_classifier.joblib")

st.set_page_config(
    page_title="Financial Complaint Classifier",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def get_model():
    """Load the trained model only once."""
    return load_model(str(MODEL_PATH))


st.title("Financial Complaint Classifier")

st.write(
    "Classify consumer financial complaints into "
    "11 product categories using Machine Learning."
)

st.divider()

complaint = st.text_area(
    "Enter your financial complaint",
    placeholder=(
        "Example: My credit card was charged twice, "
        "and the bank has not resolved the issue."
    ),
    height=180,
)

if st.button(
    "Classify Complaint",
    type="primary",
    use_container_width=True,
):
    if not complaint.strip():
        st.warning("Please enter a complaint first.")

    elif not MODEL_PATH.is_file():
        st.error(
            "Trained model not found. "
            "Please train and save the model first."
        )

    else:
        try:
            model = get_model()

            predicted_category = predict_complaint(
                model,
                complaint.strip(),
            )

            st.success("Complaint classified successfully!")

            st.subheader("Predicted Product Category")
            st.info(str(predicted_category))

        except Exception:
            st.error(
                "Unable to classify this complaint. "
                "Please check the model and application setup."
            )

st.divider()

st.caption(
    "Powered by TF-IDF and Logistic Regression. "
    "Predictions are informational and should not "
    "replace human review."
)
