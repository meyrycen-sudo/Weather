app_py_content = """import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Weather Prediction App", layout="centered")
st.title("⛅ Weather Anomaly & Prediction App")

# Load the model and preprocessing dict
pkl_path = 'app.pkl'
if not os.path.exists(pkl_path):
    st.error(f"Model file '{pkl_path}' not found! Please run the training code first.")
else:
    with open(pkl_path, 'rb') as f:
        artifacts = pickle.load(f)

    model = artifacts['model']
    feature_names = artifacts['feature_names']
    label_encoders = artifacts['label_encoders']

    st.subheader("Please answer the following Yes/No questions to predict weather conditions:")
    user_inputs = {}

    # Generate inputs dynamically as Yes/No binary questions
    for feature in feature_names:
        # Friendly display name for features
        display_name = feature.replace('_', ' ').title()
        question = f"Is there high/active '{display_name}'?"

        if feature in label_encoders and feature != 'target_encoder':
            # Categorical feature input
            options = list(label_encoders[feature].classes_)
            selected_opt = st.selectbox(f"{display_name}", options=options)
            user_inputs[feature] = label_encoders[feature].transform([selected_opt])[0]
        else:
            # Numeric features are asked as simple Yes/No (mapping Yes -> 1.0, No -> 0.0)
            choice = st.selectbox(question, options=["No", "Yes"])
            user_inputs[feature] = 1.0 if choice == "Yes" else 0.0

    if st.button("Predict Weather"):
        input_df = pd.DataFrame([user_inputs])[feature_names]
        prediction = model.predict(input_df)[0]

        st.write("--- ")
        # Decode target variable if encoder exists
        if 'target_encoder' in label_encoders:
            decoded_pred = label_encoders['target_encoder'].inverse_transform([prediction])[0]
            st.success(f"Predicted Weather Status: **{decoded_pred}**")
        else:
            st.success(f"Predicted Weather Class: **{prediction}**")
"""
print("Saved updated /content/app.py with Yes/No interactive selection inputs")
