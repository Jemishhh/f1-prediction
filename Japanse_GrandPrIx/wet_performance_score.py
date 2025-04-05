import fastf1
import pandas as pd

# Enable FastF1 caching
fastf1.Cache.enable_cache("./f1_cache")

# Load the 2023 Canadian Grand Prix (wet race)
session_2023 = fastf1.get_session(2023, "Canada", "R")
session_2023.load()

# Load the 2022 Canadian Grand Prix (dry race)
session_2022 = fastf1.get_session(2022, "Canada", "R")
session_2022.load()

# Extract lap times for both races
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()

# Drop NaN values (in case there are missing laps)
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)

# Calculate the average lap time for each driver in both races
avg_lap_2023 = laps_2023.groupby("Driver")["LapTime"].mean().reset_index()
avg_lap_2022 = laps_2022.groupby("Driver")["LapTime"].mean().reset_index()

# Convert timedelta to seconds for comparison
avg_lap_2023["LapTime (s)"] = avg_lap_2023["LapTime"].dt.total_seconds()
avg_lap_2022["LapTime (s)"] = avg_lap_2022["LapTime"].dt.total_seconds()

# Merge the data from both races on the 'Driver' column
merged_data = pd.merge(avg_lap_2023, avg_lap_2022, on="Driver", suffixes=('_2023', '_2022'))

# Calculate the performance difference in lap times between 2023 and 2022
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]

# Calculate the percentage change in lap time between the wet and dry conditions
merged_data["PerformanceChange (%)"] = merged_data["LapTimeDiff (s)"] / merged_data["LapTime (s)_2022"] * 100

# Now, we can create a wet performance score
merged_data["WetPerformanceScore"] = 1 + (merged_data["PerformanceChange (%)"] / 100)

# Print out the wet performance scores for each driver
print("\nDriver Wet Performance Scores (2023 vs 2022):")
print(merged_data[["Driver", "WetPerformanceScore"]])
