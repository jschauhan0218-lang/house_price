import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("gold_price_model.pkl")

# Page title
st.set_page_config(page_title="Gold Price Prediction", page_icon="💰")

st.title("💰 Gold Price Prediction App")

st.write("Enter the market values below to predict the Gold Price (GLD).")

# Input fields
spx = st.number_input("SPX", value=1400.0)
uso = st.number_input("USO", value=75.0)
slv = st.number_input("SLV", value=15.0)
eur_usd = st.number_input("EUR/USD", value=1.40)

# Predict button
if st.button("Predict Gold Price"):

    input_data = pd.DataFrame({
        "SPX": [spx],
        "USO": [uso],
        "SLV": [slv],
        "EUR/USD": [eur_usd]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Gold Price (GLD): {prediction[0]:.2f}")