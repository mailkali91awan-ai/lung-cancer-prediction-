import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ===========================
# Load model and feature order
# ===========================
@st.cache_resource
def load_model_package():
    return joblib.load("lung_cancer_all_models.pkl")

model_package = load_model_package()
FEATURE_ORDER = pickle.load(open("feature_order.pkl", "rb"))

# Extract models and encoders
models = {
    "Logistic Regression": model_package["logistic_regression"],
    "Random Forest": model_package["random_forest"],
    "SVM": model_package["svm"],
}
feature_encoders = model_package.get("feature_encoders", {})
target_encoder = model_package.get("target_encoder", None)

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("🫁 Lung Cancer Risk Prediction")
st.write(
    """
This app predicts **lung cancer level** using ML models trained on the Cancer Patients & Air Pollution dataset.
"""
)

# Sidebar - choose model
st.sidebar.header("Select Model")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# ===========================
# Get user input
# ===========================
def get_user_input():
    st.header("Patient Information")

    data = {}
    # Age
    data["Age"] = st.slider("Age (years)", 20, 90, 45)
    # Gender
    data["Gender"] = st.selectbox("Gender", ["Male", "Female"])

    # All other risk factors (1-8 scale)
    col1, col2 = st.columns(2)

    features_col1 = [
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
    ]

    features_col2 = [
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

    with col1:
        for f in features_col1:
            data[f] = st.slider(f, 1, 8, 3)

    with col2:
        for f in features_col2:
            data[f] = st.slider(f, 1, 8, 3)

    df = pd.DataFrame([data])
    return df

input_df = get_user_input()
st.markdown("---")

# ===========================
# Prepare input for prediction
# ===========================
def prepare_input(df):
    # Encode categorical features
    for col, le in feature_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    # Fill missing values
    df = df.fillna(0)
    # Reorder columns to match training
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df

X_input = prepare_input(input_df)

# ===========================
# Prediction
# ===========================
if st.button("Predict"):
    try:
        pred = model.predict(X_input)[0]

        # Map to original label
        if target_encoder is not None:
            pred_label = target_encoder.inverse_transform([pred])[0]
        else:
            pred_label = str(pred)

        st.subheader("🎯 Prediction Result")
        st.success(f"**Predicted Lung Cancer Level:** {pred_label}")

        # Show probabilities if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            if target_encoder is not None:
                class_labels = target_encoder.inverse_transform(np.arange(len(proba)))
            else:
                class_labels = [str(i) for i in range(len(proba))]

            proba_df = pd.DataFrame({
                "Class": class_labels,
                "Probability": proba
            }).sort_values("Probability", ascending=False).reset_index(drop=True)

            st.subheader("📊 Prediction Probabilities")
            st.dataframe(proba_df)

    except Exception as e:
        st.error("❌ Prediction failed!")
        st.code(str(e))
else:
    st.info("Set patient information above and click **Predict** to see the result.")
