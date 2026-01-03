import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# Load the all-in-one model package
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

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.title("Lung Cancer Risk Prediction")
st.write("Predict Lung Cancer Risk Level using 3 ML Models")

# Sidebar to choose model
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Input form
st.header("Patient Information")
with st.form("patient_form"):
    Age = st.number_input("Age", min_value=1, max_value=120, value=50)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Air_Pollution = st.selectbox("Air Pollution", ["High", "Medium", "Low"])
    Alcohol_use = st.selectbox("Alcohol use", ["Yes", "No"])
    Dust_Allergy = st.selectbox("Dust Allergy", ["Yes", "No"])
    OccuPational_Hazards = st.selectbox("Occupational Hazards", ["High", "Medium", "Low"])
    Genetic_Risk = st.selectbox("Genetic Risk", ["High", "Medium", "Low"])
    chronic_Lung_Disease = st.selectbox("Chronic Lung Disease", ["Yes", "No"])
    Balanced_Diet = st.selectbox("Balanced Diet", ["Yes", "No"])
    Obesity = st.selectbox("Obesity", ["Yes", "No"])
    Smoking = st.selectbox("Smoking", ["Yes", "No"])
    Passive_Smoker = st.selectbox("Passive Smoker", ["Yes", "No"])
    Chest_Pain = st.selectbox("Chest Pain", ["Yes", "No"])
    Coughing_of_Blood = st.selectbox("Coughing of Blood", ["Yes", "No"])
    Fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    Weight_Loss = st.selectbox("Weight Loss", ["Yes", "No"])
    Shortness_of_Breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    Wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    Swallowing_Difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
    Clubbing_of_Finger_Nails = st.selectbox("Clubbing of Finger Nails", ["Yes", "No"])
    Frequent_Cold = st.selectbox("Frequent Cold", ["Yes", "No"])
    Dry_Cough = st.selectbox("Dry Cough", ["Yes", "No"])
    Snoring = st.selectbox("Snoring", ["Yes", "No"])
    
    submit = st.form_submit_button("Predict")

# ------------------------------------------
# Prediction Logic
# ------------------------------------------
if submit:
    # Create DataFrame
    patient_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Air Pollution": Air_Pollution,
        "Alcohol use": Alcohol_use,
        "Dust Allergy": Dust_Allergy,
        "OccuPational Hazards": OccuPational_Hazards,
        "Genetic Risk": Genetic_Risk,
        "chronic Lung Disease": chronic_Lung_Disease,
        "Balanced Diet": Balanced_Diet,
        "Obesity": Obesity,
        "Smoking": Smoking,
        "Passive Smoker": Passive_Smoker,
        "Chest Pain": Chest_Pain,
        "Coughing of Blood": Coughing_of_Blood,
        "Fatigue": Fatigue,
        "Weight Loss": Weight_Loss,
        "Shortness of Breath": Shortness_of_Breath,
        "Wheezing": Wheezing,
        "Swallowing Difficulty": Swallowing_Difficulty,
        "Clubbing of Finger Nails": Clubbing_of_Finger_Nails,
        "Frequent Cold": Frequent_Cold,
        "Dry Cough": Dry_Cough,
        "Snoring": Snoring
    }])

    # Encode categorical features
    for col, le in feature_encoders.items():
        patient_data[col] = le.transform(patient_data[col])

    # Predict
    pred_encoded = selected_model.predict(patient_data)
    pred_label = target_encoder.inverse_transform(pred_encoded)

    st.success(f"Predicted Lung Cancer Level ({selected_model_name}): {pred_label[0]}")
