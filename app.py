import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load the all-in-one model package
# -------------------------------
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

# -------------------------------
# Sidebar: Model selection
# -------------------------------
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# -------------------------------
# Dynamic input form
# -------------------------------
st.header("Patient Information")
patient_inputs = {}

with st.form("patient_form"):
    for col in feature_encoders.keys():
        if col.lower() == "age":  # numeric input
            patient_inputs[col] = st.number_input(col, min_value=1, max_value=120, value=50)
        else:
            # Use the classes from the saved LabelEncoder
            categories = list(feature_encoders[col].classes_)
            patient_inputs[col] = st.selectbox(col, categories)
    
    submit = st.form_submit_button("Predict")

# -------------------------------
# Prediction logic
# -------------------------------
if submit:
    # Convert input dictionary to DataFrame
    patient_df = pd.DataFrame([patient_inputs])

    # Encode categorical features safely
    for col, le in feature_encoders.items():
        # Only encode if the column exists in the DataFrame
        if col in patient_df.columns:
            try:
                patient_df[col] = le.transform(patient_df[col])
            except Exception as e:
                st.error(f"Error encoding column '{col}': {e}")
        else:
            st.warning(f"Column '{col}' missing in input data!")

    # Predict
    try:
        pred_encoded = selected_model.predict(patient_df)
        pred_label = target_encoder.inverse_transform(pred_encoded)
        st.success(f"Predicted Lung Cancer Level ({selected_model_name}): {pred_label[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
