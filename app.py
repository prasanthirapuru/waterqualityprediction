import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import gdown

# Download model files if not exist
MODEL_FILE = "pollution_model.pkl"
MODEL_COLS_FILE = "model_columns.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1np0Xo-di9083ehxeKC4vRAx3gYZ-98OP"
MODEL_COLS_URL = "https://drive.google.com/uc?id=YOUR_MODEL_COLUMNS_FILE_ID"

if not os.path.exists(MODEL_FILE):
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

if not os.path.exists(MODEL_COLS_FILE):
    gdown.download(MODEL_COLS_URL, MODEL_COLS_FILE, quiet=False)

# Load model
model = joblib.load(MODEL_FILE)
model_cols = joblib.load(MODEL_COLS_FILE)

# Streamlit UI
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f'{p}: {val:.2f}')
