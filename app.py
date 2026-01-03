import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Models and Encoders
# -----------------------------
with open("lung_cancer_models.pkl", "rb") as f:
    data = pickle.load(f)

models = data['models']             # Dictionary of models
feature_encoders = data['encoders'] # Dictionary of LabelEncoders for features
target_encoder = data['target_encoder']  # LabelEncoder for target

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Lung Cancer Level Prediction")
st.write("Enter patient information below:")

# Example: create user input fields
# Replace these with actual features in your dataset
patient_inputs = {
    'Age': st.number_input("Age", min_value=1, max_value=120, value=50),
    'Gender': st.selectbox("Gender", ["Male", "Female"]),
    'Smoking': st.selectbox("Smoking", ["Yes", "No"]),
    'Air_Pollution': st.selectbox("Air Pollution", ["High", "Low"]),
    'Genetic_Risk': st.selectbox("Genetic Risk", ["High", "Low"]),
    'Obesity': st.selectbox("Obesity", ["Yes", "No"])
}

# Convert user input to DataFrame
patient_data = pd.DataFrame([patient_inputs])

# -----------------------------
# Encode categorical features safely
# -----------------------------
for col, le in feature_encoders.items():
    if col == 'Level':  # Skip target
        continue
    if col in patient_data.columns:
        patient_data[col] = le.transform(patient_data[col])

# -----------------------------
# Model Selection
# -----------------------------
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):
    try:
        pred_encoded = selected_model.predict(patient_data)
        pred_label = target_encoder.inverse_transform(pred_encoded)
        st.success(f"Predicted Lung Cancer Level ({selected_model_name}): {pred_label[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
