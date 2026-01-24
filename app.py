import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load the saved models
# -------------------------
@st.cache_resource
def load_model_package():
    return joblib.load("lung_cancer_all_models.pkl")

model_package = load_model_package()

# Expected keys from your training script
models = {
    "Logistic Regression": model_package.get("logistic_regression"),
    "Random Forest": model_package.get("random_forest"),
    "SVM": model_package.get("svm"),
}
feature_encoders = model_package.get("feature_encoders", {})
target_encoder = model_package.get("target_encoder", None)

# Features used in the Cancer Patients & Air Pollution dataset
FEATURES = [
    "Age",
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
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

st.title("Lung Cancer Risk Prediction")
st.write(
    """
This app uses machine learning models (Logistic Regression, Random Forest, and SVM)  
trained on the **Cancer Patients and Air Pollution** dataset to predict lung cancer level.
"""
)

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.write("Fill in the patient information on the main page and click **Predict**.")


def get_user_input():
    st.header("Patient Information")

    st.markdown(
        """
- **Age**: actual age in years  
- All other risk factors are on a **1–8 scale**  
  - 1 = very low / no severity  
  - 8 = very high severity
"""
    )

    data = {}

    # Age
    data["Age"] = st.slider("Age (years)", 20, 90, 50, step=1)

    col1, col2 = st.columns(2)

    with col1:
        data["Air Pollution"] = st.slider("Air Pollution", 1, 8, 4, step=1)
        data["Alcohol use"] = st.slider("Alcohol use", 1, 8, 2, step=1)
        data["Dust Allergy"] = st.slider("Dust Allergy", 1, 8, 3, step=1)
        data["OccuPational Hazards"] = st.slider("OccuPational Hazards", 1, 8, 3, step=1)
        data["Genetic Risk"] = st.slider("Genetic Risk", 1, 8, 4, step=1)
        data["chronic Lung Disease"] = st.slider("chronic Lung Disease", 1, 8, 2, step=1)
        data["Balanced Diet"] = st.slider("Balanced Diet", 1, 8, 4, step=1)
        data["Obesity"] = st.slider("Obesity", 1, 8, 3, step=1)
        data["Smoking"] = st.slider("Smoking", 1, 8, 4, step=1)
        data["Passive Smoker"] = st.slider("Passive Smoker", 1, 8, 2, step=1)
        data["Chest Pain"] = st.slider("Chest Pain", 1, 8, 3, step=1)

    with col2:
        data["Coughing of Blood"] = st.slider("Coughing of Blood", 1, 8, 1, step=1)
        data["Fatigue"] = st.slider("Fatigue", 1, 8, 3, step=1)
        data["Weight Loss"] = st.slider("Weight Loss", 1, 8, 2, step=1)
        data["Shortness of Breath"] = st.slider("Shortness of Breath", 1, 8, 3, step=1)
        data["Wheezing"] = st.slider("Wheezing", 1, 8, 3, step=1)
        data["Swallowing Difficulty"] = st.slider("Swallowing Difficulty", 1, 8, 1, step=1)
        data["Clubbing of Finger Nails"] = st.slider("Clubbing of Finger Nails", 1, 8, 1, step=1)
        data["Frequent Cold"] = st.slider("Frequent Cold", 1, 8, 3, step=1)
        data["Dry Cough"] = st.slider("Dry Cough", 1, 8, 3, step=1)
        data["Snoring"] = st.slider("Snoring", 1, 8, 3, step=1)

    # Build DataFrame with columns in the same order as training
    df_input = pd.DataFrame([data], columns=FEATURES)
    return df_input


input_df = get_user_input()

st.markdown("---")

if st.button("Predict"):
    X_input = input_df.copy()

    # If you ever had categorical feature encoders, apply them here
    # (for this dataset, features are numeric so this usually does nothing)
    for col, le in feature_encoders.items():
        if col in X_input.columns:
            X_input[col] = le.transform(X_input[col])

    # Predict
    pred = model.predict(X_input)[0]

    # Map prediction back to original label if target encoder exists
    if target_encoder is not None:
        try:
            pred_label = target_encoder.inverse_transform([pred])[0]
        except Exception:
            pred_label = str(pred)
    else:
        pred_label = str(pred)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Lung Cancer Level:** {pred_label}")

    # Show probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]

        if target_encoder is not None:
            try:
                class_labels = target_encoder.inverse_transform(
                    np.arange(len(proba))
                )
            except Exception:
                class_labels = [str(i) for i in range(len(proba))]
        else:
            class_labels = [str(i) for i in range(len(proba))]

        proba_df = (
            pd.DataFrame({"Class": class_labels, "Probability": proba})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        st.subheader("Prediction Probabilities")
        st.dataframe(proba_df)
else:
    st.info("Set the patient parameters above and click **Predict** to see the result.")
