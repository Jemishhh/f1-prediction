# prediction.py

import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('./f1_cache')

# Load 2024 Chinese GP Race Session
session_2024 = fastf1.get_session(2024, "China", 'R')
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Get average sector times per driver
avg_sector_times = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Simulated 2025 Chinese GP Qualifying Data
qualifying_2025 = pd.DataFrame({
    'Driver': [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
        "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
        "Esteban Ocon", "Nico Hulkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz",
        "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"
    ],
    'Qualifying Time (s)': [
        91.328, 91.442, 91.505, 91.537, 91.648,
        91.710, 91.799, 91.826, 92.203, 92.269,
        92.190, 92.212, 92.270, 92.321, 92.397,
        92.539, 92.582, 92.621, 92.684, 92.719
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

# Prepare training dataset
merged_data = qualifying_2025.merge(avg_sector_times, left_on="DriverCode", right_on="Driver", suffixes=("", "_avg"))
X = merged_data[["Qualifying Time (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["DriverCode"].values).values

if X.shape[0] == 0:
    raise ValueError("Feature matrix is empty after preprocessing. Check data sources.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)


# Merge qualifying_2025 with average sector times
predict_data = qualifying_2025.merge(avg_sector_times, left_on="DriverCode", right_on="Driver", how="left", suffixes=("", "_avg"))

# Fill missing sector times with mean sector times from training data
sector_means = X[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean()
predict_data[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]] = predict_data[
    ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
].fillna(sector_means)

# Predict race times
predict_X = predict_data[["Qualifying Time (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
predicted_lap_times = model.predict(predict_X)

# Attach predictions to DataFrame
predict_data["PredictedRaceTime (s)"] = predicted_lap_times

# Sort and display results using the correct column for driver names
predict_data = predict_data.sort_values(by='PredictedRaceTime (s)').reset_index(drop=True)

print("\nPredicted 2025 Chinese GP Race Times:")
print(predict_data[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Error (MAE): {mae:.2f} seconds")
print(f"\n🏁 Predicted Winner: {predict_data.iloc[0]['Driver']}")
