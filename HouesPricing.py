# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/user/Desktop/House_pricing/housing_price.csv")

print("Missing Values:\n", df.isnull().sum())

# Detecting outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((df < lower_bound) | (df > upper_bound)).sum()
print("\nOutliers detected per column:\n", outliers)

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

df["total_rooms"] = winsorize(df["total_rooms"], limits=[0.01, 0.01])
df["total_bedrooms"] = winsorize(df["total_bedrooms"], limits=[0.01, 0.01])
df["population"] = winsorize(df["population"], limits=[0.01, 0.01])
df["households"] = winsorize(df["households"], limits=[0.01, 0.01])
df["median_income"] = winsorize(df["median_income"], limits=[0.01, 0.01])
df["median_house_value"] = winsorize(df["median_house_value"], limits=[0.01, 0.01])

df_cleaned = df.drop(columns=["total_bedrooms", "households"])

# Standardize features (
scaler = StandardScaler()
features = ["total_rooms", "population", "median_income", "longitude", "latitude", "housing_median_age"]
df_cleaned[features] = scaler.fit_transform(df_cleaned[features])

X = df_cleaned.drop(columns=["median_house_value"])
y = df_cleaned["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\nModel Performance:")
print("R² Score:", round(r2, 3))
print("RMSE:", round(rmse, 2))

comparison = pd.DataFrame({'Actual': y_test[:10], 'Predicted': y_pred[:10]})
print("\nActual vs Predicted House Prices:")
print(comparison)

with open("model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)