# app.py
import streamlit as st
import pandas as pd
import numpy as np
from data_processing import load_and_preprocess_data
from models import build_attention_model
import tensorflow as tf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')

st.title("DON Concentration Prediction")
st.write("Upload a hyperspectral CSV file to predict DON levels in corn samples.")

# Load pre-trained model
model = build_attention_model(448)  # Adjust input_dim to match your data
try:
    model.load_weights('attention_model_weights.weights.h5')  # Updated to match saved filename
    logging.info("Model weights loaded successfully.")
except FileNotFoundError:
    st.error("Weights file 'attention_model_weights.weights.h5' not found. Please train the model first.")
    logging.error("Weights file not found.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        X_new = df.drop(columns=['hsi_id', 'vomitoxin_ppb']).values if 'vomitoxin_ppb' in df.columns else df.values
        _, _, scaler = load_and_preprocess_data('hyperspectral_data.csv')  # Load scaler from training data
        X_new_scaled = scaler.transform(X_new)

        # Predict
        predictions = model.predict(X_new_scaled).flatten()
        st.write("Predicted DON Concentrations (ppb):")
        st.write(predictions)

        # Visualization
        st.line_chart(np.mean(X_new_scaled, axis=0), use_container_width=True)
        logging.info("Prediction successful.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        logging.error(f"Prediction error: {e}")
