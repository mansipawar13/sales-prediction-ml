import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("adidas_rf_model.pkl")
model_features = joblib.load("model_features.pkl")

st.title("Adidas Sales Method Prediction")

st.write("Enter product sales details:")

# User inputs
price = st.number_input("Price per Unit")
units = st.number_input("Units Sold")
margin = st.number_input("Operating Margin")

# Create input dataframe
input_data = pd.DataFrame({
    "Price per Unit": [price],
    "Units Sold": [units],
    "Operating Margin": [margin]
})

# Match training feature structure
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict Sales Method"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales Method: {prediction[0]}")