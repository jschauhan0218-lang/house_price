import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("gender_model.pkl")

st.set_page_config(
    page_title="Gender Classification",
    page_icon="👤"
)

st.title("👤 Gender Classification App")

st.write("Enter the details below to predict gender.")

# Load dataset to get feature names
df = pd.read_csv("gender_classification_v7.csv")

# Remove target column
target_column = "gender"      # Change this if your target column has a different name

X = df.drop(columns=[target_column])

inputs = {}

st.subheader("Input Features")

for col in X.columns:

    if X[col].dtype == "object":
        values = list(df[col].unique())
        inputs[col] = st.selectbox(col, values)

    else:
        inputs[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )

input_df = pd.DataFrame([inputs])

# Encode categorical variables
for col in input_df.columns:
    if input_df[col].dtype == object:
        mapping = {v: i for i, v in enumerate(df[col].unique())}
        input_df[col] = input_df[col].map(mapping)

if st.button("Predict"):

    prediction = model.predict(input_df)

    st.success(f"Predicted Gender: {prediction[0]}")