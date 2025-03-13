import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
file_path = os.path.join(os.getcwd(), "housing_price.csv")
df = pd.read_csv(file_path)


# Select relevant features and target variable
features = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
target = "median_house_value"

X = df[features]
y = df[target]

# Handle missing values (if any)
X.fillna(X.mean())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




with open("model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

# Streamlit App
st.title("🏠 House Price Prediction App")
st.write("Enter details below to predict the median house price:")

# Layout for input parameters
col1, col2, col3 = st.columns(3)
with col1:
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=3)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=50000, value=5)
with col2:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=10000, value=1)
    population = st.number_input("Population", min_value=1, max_value=50000, value=3)
with col3:
    households = st.number_input("Households", min_value=1, max_value=10000, value=10)
    median_income = st.number_input("Median Income", min_value=0.1, max_value=20.0, value=5.0)

# Make prediction
# Define user input before button click
user_input = np.array([[housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])

# Display user input parameters as a table
comparison = pd.DataFrame({"Feature": features, "Input Value": user_input.flatten()})
st.write("### User Input Parameters")
st.table(comparison)

# Make prediction when the button is clicked
if st.button("Predict House Price"):
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    st.success(f"🏡 Estimated Median House Value: ${prediction[0]:,.2f}")


    
