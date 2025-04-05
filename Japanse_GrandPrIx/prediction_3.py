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

# Parse 2025 qualifying data
qual_data = [
    ["Max Verstappen", "Red Bull", "1:26.983"],
    ["Lando Norris", "McLaren", "+0.012"],
    ["Oscar Piastri", "McLaren", "+0.044"],
    ["Charles Leclerc", "Ferrari", "+0.316"],
    ["George Russell", "Mercedes", "+0.335"],
    ["Kimi Antonelli", "Mercedes", "+0.572"],
    ["Isack Hadjar", "Racing Bulls", "+0.586"],
    ["Lewis Hamilton", "Ferrari", "+0.627"],
    ["Alex Albon", "Williams", "+0.632"],
    ["Oliver Bearman", "Haas", "+0.884"],
    ["Pierre Gasly", "Alpine", "1:27.822"],
    ["Carlos Sainz", "Williams", "1:27.836"],
    ["Fernando Alonso", "Aston Martin", "1:27.897"],
    ["Liam Lawson", "Racing Bulls", "1:27.906"],
    ["Yuki Tsunoda", "Red Bull", "1:28.000"],
    ["Nico Hulkenberg", "Sauber", "1:28.570"],
    ["Gabriel Bortoleto", "Sauber", "1:28.622"],
    ["Esteban Ocon", "Haas", "1:28.696"],
    ["Jack Doohan", "Alpine", "1:28.877"],
    ["Lance Stroll", "Aston Martin", "1:29.271"]
]

pole_time = pd.to_timedelta("0:01:26.983").total_seconds()
qualifying_parsed = []
for i, row in enumerate(qual_data):
    driver, team, time_str = row
    if "+" in time_str:
        time = pole_time + float(time_str.replace("+", ""))
    else:
        time = pd.to_timedelta("0:" + time_str).total_seconds()
    qualifying_parsed.append([driver, team, i + 1, time, time - pole_time])

qualifying_2025 = pd.DataFrame(qualifying_parsed, columns=["Driver", "Team", "QualifyingPosition", "Qualifying Time (s)", "GapFromPole (s)"])

# Driver mapping
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW",
    "Kimi Antonelli": "ANT", "Alex Albon": "ALB"
}
qualifying_2025['DriverCode'] = qualifying_2025['Driver'].map(driver_mapping)

# Driver median lap times
driver_medians = laps_2024.groupby("Driver")["LapTime (s)"].median()
laps_2024['Team'] = laps_2024['Driver'].map(lambda x: qualifying_2025.loc[qualifying_2025['DriverCode'] == x, 'Team'].values[0] if x in qualifying_2025['DriverCode'].values else None)
team_medians = laps_2024.groupby("Team")["LapTime (s)"].median()
qualifying_2025['MedianLapTime (s)'] = qualifying_2025['DriverCode'].map(driver_medians)
qualifying_2025['TeamMedian (s)'] = qualifying_2025['Team'].map(team_medians)
qualifying_2025['FinalMedian (s)'] = qualifying_2025['MedianLapTime (s)'].fillna(qualifying_2025['TeamMedian (s)'])

# Consistency (std deviation of lap times)
driver_consistency = laps_2024.groupby("Driver")["LapTime (s)"].std()
qualifying_2025['Consistency (s)'] = qualifying_2025['DriverCode'].map(driver_consistency)

# Merge with sector data
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="inner", suffixes=("", "_avg"))

# Wet performance factor
wet_factors = {
    "Alexander Albon": 0.978120, "Fernando Alonso": 0.972655, "Valtteri Bottas": 0.982052,
    "Pierre Gasly": 0.978832, "Lewis Hamilton": 0.976464, "Charles Leclerc": 0.975862,
    "Kevin Magnussen": 0.989983, "Lando Norris": 0.978179, "Esteban Ocon": 0.981810,
    "Sergio Perez": 0.998904, "George Russell": 0.968678, "Carlos Sainz Jr.": 0.978754,
    "Lance Stroll": 0.979857, "Yuki Tsunoda": 0.996338, "Max Verstappen": 0.975196,
    "Guanyu Zhou": 0.987774, "Oscar Piastri": 0.980123
}
merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(wet_factors)

# Weather forecast
API_KEY = "your_api_token"
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
X = merged_data[["Qualifying Time (s)", "GapFromPole (s)", "QualifyingPosition", "FinalMedian (s)", "Consistency (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor"]]

# Target variable: use median lap time
y = merged_data["FinalMedian (s)"]

# Train-test split and model
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict
predicted_race_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_race_times

# Output sorted predictions
result_df = merged_data[["Driver", "PredictedRaceTime (s)"]].sort_values("PredictedRaceTime (s)")
print("\nPredicted 2025 Japanese GP Race Winner (Approach 2):")
print(result_df)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Error (MAE): {mae:.2f} seconds")
print(f"\nPredicted Winner: {result_df.iloc[0]['Driver']}")
