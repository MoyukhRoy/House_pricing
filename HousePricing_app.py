import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Model file not found! Please upload model.pkl to the correct directory.")
    st.stop()

st.title("🏠 House Price Prediction App")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=30)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=50000, value=5000)
with col2:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=10000, value=1000)
    population = st.number_input("Population", min_value=1, max_value=50000, value=3000)
with col3:
    households = st.number_input("Households", min_value=1, max_value=10000, value=1000)
    median_income = st.number_input("Median Income", min_value=0.1, max_value=20.0, value=5.0)

# Make prediction
if st.button("Predict House Price"):
    user_input = np.array([[housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
    prediction = model.predict(user_input)

    st.success(f"🏡 Estimated Median House Value: ${int(prediction[0]):,}")

    # Display input values
    input_data = pd.DataFrame({
        "Feature": ["Housing Median Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Median Income"],
        "Input Value": [int(housing_median_age), int(total_rooms), int(total_bedrooms), int(population), int(households), round(median_income, 2)]
    })
    st.write("### Entered Values")
    st.table(input_data)
