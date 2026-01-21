import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Load model package
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lung_cancer_all_models.pkl")

model_package = joblib.load(MODEL_PATH)

# Unpack everything
feature_encoders = model_package["feature_encoders"]
target_encoder = model_package["target_encoder"]

models = {
    "Logistic Regression": model_package["logistic_regression"],
    "Random Forest": model_package["random_forest"],
    "Support Vector Machine": model_package["svm"],
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Lung Cancer Prediction App")

st.sidebar.header("Input Patient Data")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 1, 120, 30)
    Smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
    Yellow_Fingers = st.sidebar.selectbox("Yellow Fingers", ["Yes", "No"])
    Anxiety = st.sidebar.selectbox("Anxiety", ["Yes", "No"])
    Peer_Pressure = st.sidebar.selectbox("Peer Pressure", ["Yes", "No"])
    Chronic_Disease = st.sidebar.selectbox("Chronic Disease", ["Yes", "No"])
    Fatigue = st.sidebar.selectbox("Fatigue", ["Yes", "No"])
    Allergy = st.sidebar.selectbox("Allergy", ["Yes", "No"])
    Wheezing = st.sidebar.selectbox("Wheezing", ["Yes", "No"])
    Alcohol_Consuming = st.sidebar.selectbox("Alcohol Consuming", ["Yes", "No"])
    Coughing = st.sidebar.selectbox("Coughing", ["Yes", "No"])
    Shortness_of_Breath = st.sidebar.selectbox("Shortness of Breath", ["Yes", "No"])
    Swallowing_Difficulty = st.sidebar.selectbox("Swallowing Difficulty", ["Yes", "No"])
    Chest_Pain = st.sidebar.selectbox("Chest Pain", ["Yes", "No"])

    data = {
        "Gender": Gender,
        "Age": Age,
        "Smoking": Smoking,
        "Yellow_Fingers": Yellow_Fingers,
        "Anxiety": Anxiety,
        "Peer_Pressure": Peer_Pressure,
        "Chronic_Disease": Chronic_Disease,
        "Fatigue": Fatigue,
        "Allergy": Allergy,
        "Wheezing": Wheezing,
        "Alcohol_Consuming": Alcohol_Consuming,
        "Coughing": Coughing,
        "Shortness_of_Breath": Shortness_of_Breath,
        "Swallowing_Difficulty": Swallowing_Difficulty,
        "Chest_Pain": Chest_Pain
    }

    return pd.DataFrame(data, index=[0])


patient_data = user_input_features()

# -----------------------------
# Encode categorical columns
# -----------------------------
for col, encoder in feature_encoders.items():
    if col in patient_data:
        patient_data[col] = encoder.transform(patient_data[col])

# -----------------------------
# Select model
# -----------------------------
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[model_choice]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    pred = selected_model.predict(patient_data)
    pred_label = target_encoder.inverse_transform(pred)

    st.subheader("Prediction Result")
    if pred_label[0] == "Yes":
        st.error("⚠️ Patient is likely to have Lung Cancer.")
    else:
        st.success("✅ Patient is unlikely to have Lung Cancer.")

    if hasattr(selected_model, "predict_proba"):
        st.subheader("Prediction Probability")
        st.write(selected_model.predict_proba(patient_data))
