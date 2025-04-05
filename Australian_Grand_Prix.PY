# prediction.py

import os
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Enable FastF1 caching
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# Load 2024 Australian GP Race Session
session_2024 = fastf1.get_session(2024, 3, 'R')
session_2024.load()

# Extract lap data
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()


# Simulated 2025 Qualifying Data (replace with real data if available)
qualifying_2025 = pd.DataFrame({
    'Driver': [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
        "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
        "Esteban Ocon", "Nico Hulkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz",
        "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"
    ],
    'Qualifying Time (s)': [
        90.641, 90.723, 90.793, 90.817, 90.927,
        91.021, 91.079, 91.103, 91.638, 91.706,
        91.625, 91.632, 91.688, 91.773, 91.840,
        91.992, 92.018, 92.092, 92.141, 92.174
    ]
})

# Driver mapping (2025 names to 3-letter codes)
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO"
}

qualifying_2025['DriverCode'] = qualifying_2025['Driver'].map(driver_mapping)

# Merge qualifying with 2024 race data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Use only QualifyingTime(s) as feature
X = merged_data[['Qualifying Time (s)']]
y = merged_data['LapTime (s)']

if X.shape[0] == 0:
    raise ValueError("Feature matrix is empty after preprocessing. Check data sources.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

predicted_lap_times = model.predict(qualifying_2025[["Qualifying Time (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers
qualifying_2025 = qualifying_2025.sort_values(by='PredictedRaceTime (s)').reset_index(drop=True)

# Show predictions
print("\nPredicted 2025 Australian GP Race Times:")
print(qualifying_2025[['Driver', 'PredictedRaceTime (s)']])

# Final Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Error (MAE): {mae:.2f} seconds")
print(f"\n🏁 Predicted Winner: {qualifying_2025.iloc[0]['Driver']}")
