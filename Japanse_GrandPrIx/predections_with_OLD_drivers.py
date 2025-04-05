import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache('./f1_cache')

# Load 2024 Japanese GP Race Session
session_2024 = fastf1.get_session(2024, "Japan", 'R')
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Updated 2025 Japanese GP Official Qualifying Times
qualifying_2025 = pd.DataFrame({
    'Driver': [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
        "Charles Leclerc", "Yuki Tsunoda", "Alexander Albon",
        "Esteban Ocon", "Nico Hulkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz",
        "Pierre Gasly"
    ],
    'Qualifying Time (s)': [
        90.641, 90.723, 90.793, 90.817, 90.927,
        91.021, 91.638, 91.706,
        91.625, 91.632, 91.688, 91.773, 91.840,
        91.992
    ]
})

driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}
qualifying_2025['DriverCode'] = qualifying_2025['Driver'].map(driver_mapping)

# Merge with 2024 sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="inner", suffixes=("", "_avg"))

# Wet performance factor (static)
driver_wet_performance = {
    "Alexander Albon": 0.978120, "Fernando Alonso": 0.972655, "Valtteri Bottas": 0.982052,
    "Pierre Gasly": 0.978832, "Lewis Hamilton": 0.976464, "Charles Leclerc": 0.975862,
    "Kevin Magnussen": 0.989983, "Lando Norris": 0.978179, "Esteban Ocon": 0.981810,
    "Sergio Perez": 0.998904, "George Russell": 0.968678, "Carlos Sainz Jr.": 0.978754,
    "Lance Stroll": 0.979857, "Yuki Tsunoda": 0.996338, "Max Verstappen": 0.975196,
    "Guanyu Zhou": 0.987774, "Oscar Piastri": 0.980123
}
merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(driver_wet_performance)

# Weather API (forecast for Suzuka)
API_KEY = "7567028813d733a10acdac7f7472301b"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Suzuka,jp&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

forecast_time = "2025-04-06 14:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else None

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Feature matrix
X = merged_data[["Qualifying Time (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor"]]

# Target variable
avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
y = merged_data["DriverCode"].map(avg_lap_times)

# Impute and split
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

# Model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict
predicted_race_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_race_times

# Safely assign back to qualifying_2025
pred_df = merged_data[["Driver", "PredictedRaceTime (s)"]]
qualifying_2025 = qualifying_2025.merge(pred_df, on="Driver", how="left")

# Sort only those with predictions
qualifying_2025_sorted = qualifying_2025.dropna(subset=["PredictedRaceTime (s)"]).sort_values("PredictedRaceTime (s)")

# Output
print("\nPredicted 2025 Japanese GP Race Winner (Approach 1s):")
print(qualifying_2025_sorted[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Error (MAE): {mae:.2f} seconds")
print(f"\nPredicted Winner: {qualifying_2025_sorted.iloc[0]['Driver']}")
