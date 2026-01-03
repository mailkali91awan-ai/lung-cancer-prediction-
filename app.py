import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# -----------------------------
# Suppress warnings (optional)
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# Load saved models & encoders
# -----------------------------
with open("models/feature_encoders.pkl", "rb") as f:
    feature_encoders = pickle.load(f)

with open("models/logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("models/decision_tree_model.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

with open("models/random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

with open("models/svc_model.pkl", "rb") as f:
    svc_model = pickle.load(f)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "SVC": svc_model
}

# -----------------------------
# Streamlit App UI
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
    
    features = pd.DataFrame(data, index=[0])
    return features

patient_data = user_input_features()

# -----------------------------
# Encode categorical features safely
# -----------------------------
for col, le in feature_encoders.items():
    if col in patient_data.columns:
        patient_data[col] = le.transform(patient_data[col])

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[model_choice]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    pred = selected_model.predict(patient_data)
    pred_proba = selected_model.predict_proba(patient_data) if hasattr(selected_model, "predict_proba") else None
    
    st.subheader("Prediction Result")
    if pred[0] == 1:
        st.error("⚠️ Patient is likely to have Lung Cancer.")
    else:
        st.success("✅ Patient is unlikely to have Lung Cancer.")
    
    if pred_proba is not None:
        st.subheader("Prediction Probability")
        st.write(pd.DataFrame(pred_proba, columns=selected_model.classes_))
