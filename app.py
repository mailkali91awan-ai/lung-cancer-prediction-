import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load saved models & encoders
# -------------------------
@st.cache_resource
def load_model_package():
    return joblib.load("lung_cancer_all_models.pkl")

model_package = load_model_package()

models = {
    "Logistic Regression": model_package.get("logistic_regression"),
    "Random Forest": model_package.get("random_forest"),
    "SVM": model_package.get("svm"),
}

feature_encoders = model_package.get("feature_encoders", {})
target_encoder = model_package.get("target_encoder", None)

# -------------------------
# MUST match training order
# -------------------------
FEATURES = [
    "Age",
    "Gender",
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
]

# -------------------------
# UI Settings
# -------------------------
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

st.title("🔍 Lung Cancer Risk Prediction")
st.write(
    """
This machine learning app predicts **Lung Cancer Level**  
using models trained on the *Cancer Patients & Air Pollution* dataset.
"""
)

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# -------------------------
# User Input Form
# -------------------------
def get_user_input():
    st.header("Patient Information")

    data = {}
    data["Age"] = st.slider("Age (years)", 20, 90, 45)

    # Gender (Categorical)
    data["Gender"] = st.selectbox("Gender", ["Male", "Female"])

    col1, col2 = st.columns(2)

    with col1:
        data["Air Pollution"] = st.slider("Air Pollution", 1, 8, 4)
        data["Alcohol use"] = st.slider("Alcohol use", 1, 8, 2)
        data["Dust Allergy"] = st.slider("Dust Allergy", 1, 8, 3)
        data["OccuPational Hazards"] = st.slider("OccuPational Hazards", 1, 8, 3)
        data["Genetic Risk"] = st.slider("Genetic Risk", 1, 8, 4)
        data["chronic Lung Disease"] = st.slider("chronic Lung Disease", 1, 8, 2)
        data["Balanced Diet"] = st.slider("Balanced Diet", 1, 8, 4)
        data["Obesity"] = st.slider("Obesity", 1, 8, 3)
        data["Smoking"] = st.slider("Smoking", 1, 8, 4)
        data["Passive Smoker"] = st.slider("Passive Smoker", 1, 8, 2)
        data["Chest Pain"] = st.slider("Chest Pain", 1, 8, 3)

    with col2:
        data["Coughing of Blood"] = st.slider("Coughing of Blood", 1, 8, 1)
        data["Fatigue"] = st.slider("Fatigue", 1, 8, 3)
        data["Weight Loss"] = st.slider("Weight Loss", 1, 8, 2)
        data["Shortness of Breath"] = st.slider("Shortness of Breath", 1, 8, 3)
        data["Wheezing"] = st.slider("Wheezing", 1, 8, 3)
        data["Swallowing Difficulty"] = st.slider("Swallowing Difficulty", 1, 8, 1)
        data["Clubbing of Finger Nails"] = st.slider("Clubbing of Finger Nails", 1, 8, 1)
        data["Frequent Cold"] = st.slider("Frequent Cold", 1, 8, 3)
        data["Dry Cough"] = st.slider("Dry Cough", 1, 8, 3)
        data["Snoring"] = st.slider("Snoring", 1, 8, 3)

    df = pd.DataFrame([data], columns=FEATURES)
    return df


input_df = get_user_input()
st.markdown("---")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    X_input = input_df.copy()

    # Apply encoders (only Gender)
    for col, le in feature_encoders.items():
        if col in X_input.columns:
            X_input[col] = le.transform(X_input[col])

    # 🔥 FIX: Convert all values to numeric
    X_input = X_input.apply(pd.to_numeric, errors="coerce")

    # Predict class
    pred = model.predict(X_input)[0]

    # Decode class label
    if target_encoder is not None:
        try:
            pred_label = target_encoder.inverse_transform([pred])[0]
        except:
            pred_label = str(pred)
    else:
        pred_label = str(pred)

    st.subheader("🎯 Prediction Result")
    st.success(f"**Predicted Lung Cancer Level:** {pred_label}")

    # Probability Table (if available)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]

        if target_encoder is not None:
            try:
                class_labels = target_encoder.inverse_transform(
                    np.arange(len(proba))
                )
            except:
                class_labels = [str(i) for i in range(len(proba))]
        else:
            class_labels = [str(i) for i in range(len(proba))]

        proba_df = (
            pd.DataFrame({"Class": class_labels, "Probability": proba})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(proba_df)

else:
    st.info("Adjust the parameters above and click **Predict** to see the lung cancer risk level.")
