import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter the house details below.")

# Numeric Inputs
area = st.number_input("Area", min_value=500, value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
stories = st.number_input("Stories", min_value=1, value=2)
parking = st.number_input("Parking", min_value=0, value=1)

# Categorical Inputs
mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

# Convert to numeric
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

if furnishingstatus == "Unfurnished":
    furnishingstatus = 0
elif furnishingstatus == "Semi-Furnished":
    furnishingstatus = 1
else:
    furnishingstatus = 2

# Predict
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    prediction = model.predict(input_data)

    st.success(f"🏠 Predicted House Price: ₹ {prediction[0]:,.2f}")