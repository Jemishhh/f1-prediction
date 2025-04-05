#2nd approach

import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('./f1_cache')

# Load 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, "China", 'R')
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Team"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Average sector times by driver
avg_sector_times = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Average sector times by team
team_sector_times = laps_2024.groupby("Team")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Overall sector means
sector_means = avg_sector_times[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean()

# 2025 Qualifying data
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
    ],
    'Team': [
        "McLaren", "Mercedes", "McLaren", "Red Bull", "Ferrari",
        "Ferrari", "Racing Bulls", "Mercedes", "Racing Bulls", "Williams",
        "Haas", "Sauber", "Aston Martin", "Aston Martin", "Williams",
        "Alpine", "Haas", "Alpine", "Sauber", "Red Bull"
    ]
})

# Mapping Driver to 2024 driver codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge sector times
predict_data = qualifying_2025.merge(avg_sector_times, left_on="DriverCode", right_on="Driver", how="left", suffixes=("", "_avg"))

# Fill missing sector times using team averages
predict_data = predict_data.merge(team_sector_times, on="Team", how="left", suffixes=("", "_team"))

for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
    predict_data[col] = predict_data[col].fillna(predict_data[f"{col}_team"])
    predict_data[col] = predict_data[col].fillna(sector_means[col])

# Prepare training data (only known drivers from 2024 race)
train_data = predict_data[predict_data["DriverCode"].isin(avg_sector_times["Driver"])]
X = train_data[["Qualifying Time (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(train_data["DriverCode"].values).values

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict all drivers
predict_X = predict_data[["Qualifying Time (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
predict_data["PredictedRaceTime (s)"] = model.predict(predict_X)
predict_data = predict_data.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

# Display results
print("\n Predicted 2025 Chinese GP Race Times (Team-based Hybrid):")
print(predict_data[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel MAE: {mae:.2f} seconds")
print(f"\n Predicted Winner: {predict_data.iloc[0]['Driver']}")
