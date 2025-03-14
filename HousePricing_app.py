import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("housing_price.csv")

st.title("🏠 House Price Lookup App")
st.write("Enter details below to get an estimated house price:")

col1, col2, col3 = st.columns(3)
with col1:
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=30)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=50000, value=12)
with col2:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=10000, value=5)
    population = st.number_input("Population", min_value=1, max_value=50000, value=14)
with col3:
    households = st.number_input("Households", min_value=1, max_value=10000, value=1000)
    median_income = st.number_input("Median Income", min_value=0.1, max_value=20.0, value=5.0)

if st.button("Predicted House Price"):
    avg_price_per_room = df["median_house_value"].mean() / df["total_rooms"].mean()
    avg_price_per_bedroom = df["median_house_value"].mean() / df["total_bedrooms"].mean()
    avg_price_per_population = df["median_house_value"].mean() / df["population"].mean()
    avg_price_per_household = df["median_house_value"].mean() / df["households"].mean()
    avg_income_impact = df["median_house_value"].mean() / df["median_income"].mean()

    estimated_price = (
        (total_rooms * avg_price_per_room) +
        (total_bedrooms * avg_price_per_bedroom) +
        (population * avg_price_per_population) +
        (households * avg_price_per_household) +
        (median_income * avg_income_impact)
    )
    
    st.write(f"### Estimated House Price: ${int(estimated_price)}")
    
    input_data = pd.DataFrame({
        "Feature": ["Housing Median Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Median Income"],
        "Input Value": [housing_median_age, total_rooms, total_bedrooms, population, households, median_income]
    })
    st.table(input_data)
