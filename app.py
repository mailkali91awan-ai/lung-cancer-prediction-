import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# Load model package
# ------------------------------------------
@st.cache_data
def load_model():
    return joblib.load("lung_cancer_all_models.pkl")

model_package = load_model()
models = {
    "Logistic Regression": model_package["logistic_regression"],
    "Random Forest": model_package["random_forest"],
    "SVM": model_package["svm"]
}
feature_encoders = model_package["feature_encoders"]
target_encoder = model_package["target_encoder"]

st.title("Lung Cancer Risk Prediction")
st.write("Predict Lung Cancer Risk Level using 3 ML Models")

# Sidebar: choose model
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

st.header("Patient Information")

# ------------------------------------------
# Dynamic form based on feature_encoders keys
# ------------------------------------------
patient_inputs = {}
with st.form("patient_form"):
    for col in feature_encoders.keys():
        if col == "Age":
            patient_inputs[col] = st.number_input(col, min_value=1, max_value=120, value=50)
        else:
            # Get all possible categories from encoder
            categories = list(feature_encoders[col].classes_)
            patient_inputs[col] = st.selectbox(col, categories)
    submit = st.form_submit_button("Predict")

# ------------------------------------------
# Prediction
# ------------------------------------------
if submit:
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_inputs])

    # Encode categorical features
    for col, le in feature_encoders.items():
        patient_df[col] = le.transform(patient_df[col])

    # Predict
    pred_encoded = selected_model.predict(patient_df)
    pred_label = target_encoder.inverse_transform(pred_encoded)

    st.success(f"Predicted Lung Cancer Level ({selected_model_name}): {pred_label[0]}")
