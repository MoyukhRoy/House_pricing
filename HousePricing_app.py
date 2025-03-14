import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained model and scaler
with open("with open("/mount/src/house_pricing_-regression/model.pkl", "rb") as f: 
    model = pickle.load(f)

# Streamlit App
st.title("🏠 House Price Prediction App")
st.write("Enter details below to predict the median house price:")

# Layout for input parameters
col1, col2, col3 = st.columns(3)
with col1:
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=30, format="%d")
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=50000, value=10, format="%d")
with col2:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=10000, value=3, format="%d")
    population = st.number_input("Population", min_value=1, max_value=50000, value=15, format="%d")
with col3:
    households = st.number_input("Households", min_value=1, max_value=10000, value=5, format="%d")
    median_income = st.number_input("Median Income", min_value=0.1, max_value=20.0, value=5.0, format="%.2f")

# Make prediction
if st.button("Predict House Price"):
    user_input = np.array([[housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    st.success(f"🏡 Estimated Median House Value: ${prediction[0]:,.2f}")

    # Display input values in a table with int conversion except for Median Income
    input_data = pd.DataFrame({
        "Feature": ["Housing Median Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Median Income"],
        "Input Value": [int(user_input[0][0]), int(user_input[0][1]), int(user_input[0][2]), int(user_input[0][3]), int(user_input[0][4]), round(user_input[0][5], 2)]
    })
    st.write("### Entered Values")
    st.table(input_data)
