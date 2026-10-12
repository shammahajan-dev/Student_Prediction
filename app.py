
import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction")
st.write("Enter student details to predict placement status.")

# Model path
MODEL_PATH = Path(__file__).parent / "Model.pkl"

# Load model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception:
    st.error("Unable to load Model.pkl. Check your model file.")
    st.stop()

# Feature names from your index.html
feature_names = [
    "IQ",
    "CGPA",
    "10th Marks",
    "12th Marks",
    "Communication Skills"
]

# Verify model input count
n_features = getattr(model, "n_features_in_", None)

if n_features != len(feature_names):
    st.error(
        f"Your model expects {n_features} features, "
        f"but this app provides {len(feature_names)}. "
        "Please check your model's training features."
    )
    st.stop()

st.subheader("Student Information")

# Input fields
iq = st.number_input(
    "IQ", min_value=0, max_value=200, value=100
)

cgpa = st.number_input(
    "CGPA", min_value=0.0, max_value=10.0,
    value=7.0, step=0.1
)

tenth_marks = st.number_input(
    "10th Marks (%)", min_value=0.0,
    max_value=100.0, value=75.0, step=0.5
)

twelfth_marks = st.number_input(
    "12th Marks (%)", min_value=0.0,
    max_value=100.0, value=75.0, step=0.5
)

communication_skills = st.number_input(
    "Communication Skills",
    min_value=0.0, max_value=10.0,
    value=7.0, step=0.1
)

# Prediction
if st.button("Predict Placement", type="primary"):
    try:
        input_data = np.array([[
            iq,
            cgpa,
            tenth_marks,
            twelfth_marks,
            communication_skills
        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 Prediction: Placed")
        elif prediction[0] == 0:
            st.warning("Prediction: Not Placed")
        else:
            st.info(f"Prediction: {prediction[0]}")

    except Exception:
        st.error(
            "Prediction failed. Please check the model "
            "and the feature order used during training."
        )
        st.exception(Exception("See the error details above."))