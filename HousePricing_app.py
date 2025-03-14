import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("housing_price.csv")

st.title("🏠 House Price Lookup App")
st.write("Enter details below to find similar house prices from the dataset:")

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

# Filter dataset based on user input
filtered_df = df[
    (df['housing_median_age'] == housing_median_age) &
    (df['total_rooms'] == total_rooms) &
    (df['total_bedrooms'] == total_bedrooms) &
    (df['population'] == population) &
    (df['households'] == households) &
    (df['median_income'] == median_income)
]

# Display results
if not filtered_df.empty:
    st.write("### Matching House Prices")
    filtered_df_display = filtered_df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']]
    filtered_df_display = filtered_df_display.astype({
        "housing_median_age": int,
        "total_rooms": int,
        "total_bedrooms": int,
        "population": int,
        "households": int,
        "median_house_value": int
    })
    st.table(filtered_df_display)
else:
    estimated_price = df['median_house_value'].mean()
    st.warning("No exact match found in the dataset. Try adjusting the values.")
    st.write(f"### Estimated House Price: ${int(estimated_price)}")
