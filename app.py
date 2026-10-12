
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

# Model file path
MODEL_PATH = Path(__file__).parent / "Model.pkl"


# Load model only once
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# Load trained model
try:
    model = load_model()
except Exception as e:
    st.error("Unable to load Model.pkl. Please check the model file.")
    st.exception(e)
    st.stop()


# Get expected number of features
n_features = getattr(model, "n_features_in_", None)

if n_features is None:
    st.error("Unable to determine the number of input features.")
    st.stop()

st.subheader("Enter Student Information")

# Input fields
features = []

for i in range(n_features):
    value = st.number_input(
        f"Feature {i + 1}",
        value=0.0,
        step=1.0,
        key=f"feature_{i}"
    )
    features.append(value)

# Prediction button
if st.button("Predict Placement", type="primary"):
    try:
        input_data = np.array(
            features, dtype=float
        ).reshape(1, -1)

        prediction = model.predict(input_data)

        result = prediction[0]

        if result == 1:
            st.success("🎉 Prediction: Placed")
        elif result == 0:
            st.warning("Prediction: Not Placed")
        else:
            st.info(f"Prediction result: {result}")

    except Exception as e:
        st.error("Prediction failed. Please check the input features.")
        st.exception(e)