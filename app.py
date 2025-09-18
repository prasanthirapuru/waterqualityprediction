import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import gdown

# ------------------------------
# File setup
# ------------------------------
MODEL_FILE = "pollution_model.pkl"
MODEL_COLS_FILE = "model_columns.pkl"

# Google Drive link for model
MODEL_URL = "https://drive.google.com/uc?id=1np0Xo-di9083ehxeKC4vRAx3gYZ-98OP"

# Download model file if not available
if not os.path.exists(MODEL_FILE):
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Load model and column structure
model = joblib.load(MODEL_FILE)
model_cols = joblib.load(MODEL_COLS_FILE)

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("🌊 Water Pollutants Predictor")
st.markdown("A **Machine Learning app** to forecast water pollutant levels 🚀")

# Know About Project section
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    This project uses a **trained Machine Learning model** to predict water pollutant levels.  
    It considers **Year** and **Station ID** as input features.  

    **Predicted pollutants include:**
    - O₂ (Oxygen)  
    - NO₃ (Nitrate)  
    - NO₂ (Nitrite)  
    - SO₄ (Sulfate)  
    - PO₄ (Phosphate)  
    - Cl (Chloride)  

    ✅ **Use Case:**  
    Environmental researchers and organizations can monitor water quality trends over time to support decision-making.
    """)

# ------------------------------
# Input Parameters
# ------------------------------
st.header("🔹 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)

with col2:
    station_id = st.text_input("Enter Station ID", value='1')

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predict"):
    if not station_id:
        st.warning("⚠️ Please enter the station ID")
    else:
        # Prepare input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Prediction
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        results = {p: round(val, 2) for p, val in zip(pollutants, predicted_pollutants)}

        # Display results
        st.success(f"📊 Predicted Pollutant Levels for Station **{station_id}** in **{year_input}**")
        st.json(results)

