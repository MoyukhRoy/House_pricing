import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("housing_price.csv")

st.title("🏠 House Price Lookup App")
st.write("Enter details below to get an estimated house price:")

# Layout for input parameters
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

# Hardcoded estimation logic
estimated_price = (
    (housing_median_age * 1000) +
    (total_rooms * 0.5) +
    (total_bedrooms * 2) +
    (population * 0.3) +
    (households * 5) +
    (median_income * 10000)
)

st.write(f"### Estimated House Price: ${int(estimated_price)}")
