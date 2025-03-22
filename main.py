import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Churn Prediction", layout="centered")

@st.cache_resource
def load_model():
    model_path = "assets/lightgbm_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Train the model first.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

CATEGORICAL_MAPPINGS = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 
                      'Electronic check', 'Mailed check']
}

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

st.title("🔮 Customer Churn Prediction")

if model:
    st.subheader("Enter Customer Details")

    num_inputs = {feature: st.number_input(feature, value=0.0) for feature in NUMERIC_FEATURES}
    cat_inputs = {feature: st.selectbox(feature, CATEGORICAL_MAPPINGS[feature]) for feature in CATEGORICAL_MAPPINGS}

    if st.button("Predict"):
        input_data = {**num_inputs, **cat_inputs}
        
        for feature, categories in CATEGORICAL_MAPPINGS.items():
            input_data[feature] = categories.index(input_data[feature])
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
else:
    st.warning("Model not loaded. Please check the 'assets' directory.")
