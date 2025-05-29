import streamlit as st
import pandas as pd
import category_encoders as ce
import joblib

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Regression Predictor")

uploaded_file = st.file_uploader("Upload CSV for prediction")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Input Data", data)

    # Binary encode the new input
    data_encoded = encoder.transform(data)

    predictions = model.predict(data_encoded)
    st.write("Predictions", predictions)
