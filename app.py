import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Models and Encoders
# -----------------------------
@st.cache_data
def load_models():
    models = {}
    try:
        with open("logistic_model.pkl", "rb") as f:
            models['Logistic Regression'] = pickle.load(f)
        with open("decision_tree_model.pkl", "rb") as f:
            models['Decision Tree'] = pickle.load(f)
        with open("random_forest_model.pkl", "rb") as f:
            models['Random Forest'] = pickle.load(f)
        with open("svc_model.pkl", "rb") as f:
            models['SVC'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

@st.cache_data
def load_encoders():
    try:
        with open("feature_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        encoders = {}
    return encoders

# Load models and encoders
models = load_models()
feature_encoders = load_encoders()

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Lung Cancer Prediction App")
st.write("Fill in the patient details to predict lung cancer risk.")

# -----------------------------
# User Input
# -----------------------------
# Define input fields (update these according to your dataset)
def user_input_features():
    Age = st.number_input("Age", min_value=1, max_value=120, value=50)
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Smoking = st.selectbox("Smoking", ("Yes", "No"))
    Yellow_Fingers = st.selectbox("Yellow Fingers", ("Yes", "No"))
    Anxiety = st.selectbox("Anxiety", ("Yes", "No"))
    Peer_Pressure = st.selectbox("Peer Pressure", ("Yes", "No"))
    Chronic_Disease = st.selectbox("Chronic Disease", ("Yes", "No"))
    Fatigue = st.selectbox("Fatigue", ("Yes", "No"))
    Allergy = st.selectbox("Allergy", ("Yes", "No"))
    Wheezing = st.selectbox("Wheezing", ("Yes", "No"))
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", ("Yes", "No"))
    Coughing = st.selectbox("Coughing", ("Yes", "No"))
    Shortness_of_Breath = st.selectbox("Shortness of Breath", ("Yes", "No"))
    Swallowing_Difficulty = st.selectbox("Swallowing Difficulty", ("Yes", "No"))
    Chest_Pain = st.selectbox("Chest Pain", ("Yes", "No"))

    data = {
        "Age": Age,
        "Gender": Gender,
        "Smoking": Smoking,
        "Yellow_Fingers": Yellow_Fingers,
        "Anxiety": Anxiety,
        "Peer_Pressure": Peer_Pressure,
        "Chronic_Disease": Chronic_Disease,
        "Fatigue": Fatigue,
        "Allergy": Allergy,
        "Wheezing": Wheezing,
        "Alcohol_Consumption": Alcohol_Consumption,
        "Coughing": Coughing,
        "Shortness_of_Breath": Shortness_of_Breath,
        "Swallowing_Difficulty": Swallowing_Difficulty,
        "Chest_Pain": Chest_Pain
    }
    features = pd.DataFrame([data])
    return features

patient_data = user_input_features()

# -----------------------------
# Encode categorical features safely
# -----------------------------
for col, le in feature_encoders.items():
    # Skip target column and any missing columns
    if col == 'Level' or col not in patient_data.columns:
        continue
    patient_data[col] = le.transform(patient_data[col])

# -----------------------------
# Model Selection
# -----------------------------
st.subheader("Select Model")
model_choice = st.selectbox("Choose a model", list(models.keys()))
selected_model = models[model_choice]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        prediction = selected_model.predict(patient_data)
        prediction_proba = None
        if hasattr(selected_model, "predict_proba"):
            prediction_proba = selected_model.predict_proba(patient_data)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Patient is likely to have lung cancer.")
        else:
            st.success("Patient is unlikely to have lung cancer.")

        if prediction_proba is not None:
            st.subheader("Prediction Probability")
            st.write(pd.DataFrame(prediction_proba, columns=selected_model.classes_))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
