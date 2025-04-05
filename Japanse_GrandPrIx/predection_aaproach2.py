# This code is under development

import pandas as pd
import numpy as np
import fastf1
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Enable caching for faster data loading
fastf1.Cache.enable_cache('./f1_cache')

# ====================== DATA PREPARATION ======================

def load_qualifying_data():
    """Load and process 2025 Japanese GP qualifying data"""
    data = [
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
    
    qualifying_df = pd.DataFrame(data, columns=["Driver", "Team", "QualifyingTimeRaw"])
    
    # Convert times to seconds
    pole_time = pd.to_timedelta("0:01:26.983").total_seconds()

    def parse_qualifying_time(x, pole_time):
        if '+' in x:
            return pole_time + float(x.strip('+'))
        else:
            parts = x.split(':')
            if len(parts) == 2:
                x = '0:' + x  # make sure format is hh:mm:ss.xxx
            return pd.to_timedelta(x).total_seconds()

    qualifying_df["QualifyingTime"] = qualifying_df["QualifyingTimeRaw"].apply(
        lambda x: parse_qualifying_time(x, pole_time)
    )
    
    # Add driver codes and team strength
    driver_codes = {
        "Max Verstappen": "VER", "Lando Norris": "NOR", "Oscar Piastri": "PIA",
        "Charles Leclerc": "LEC", "George Russell": "RUS", "Kimi Antonelli": "ANT",
        "Isack Hadjar": "HAD", "Lewis Hamilton": "HAM", "Alex Albon": "ALB",
        "Oliver Bearman": "BEA", "Pierre Gasly": "GAS", "Carlos Sainz": "SAI",
        "Fernando Alonso": "ALO", "Liam Lawson": "LAW", "Yuki Tsunoda": "TSU",
        "Nico Hulkenberg": "HUL", "Gabriel Bortoleto": "BOR", "Esteban Ocon": "OCO",
        "Jack Doohan": "DOO", "Lance Stroll": "STR"
    }
    
    team_strength = {
        "Red Bull": 9, "McLaren": 8, "Ferrari": 8, "Mercedes": 7,
        "Aston Martin": 6, "Alpine": 5, "Racing Bulls": 5,
        "Williams": 4, "Haas": 3, "Sauber": 3
    }
    
    qualifying_df["DriverCode"] = qualifying_df["Driver"].map(driver_codes)
    qualifying_df["TeamStrength"] = qualifying_df["Team"].map(team_strength)
    
    return qualifying_df

def load_historical_data():
    """Load 2024 race data for sector time references"""
    session = fastf1.get_session(2024, "Japan", 'R')
    session.load()
    
    laps = session.laps.pick_quicklaps().pick_accurate()
    laps = laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time", "LapTime"]].dropna()
    
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col}_s"] = laps[col].dt.total_seconds()
    
    return laps.groupby("Driver").mean().reset_index()

def get_weather_data():
    """Fetch Suzuka weather forecast"""
    API_KEY = "7567028813d733a10acdac7f7472301b"
    url = f"https://api.openweathermap.org/data/2.5/forecast?q=Suzuka,jp&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        weather_data = response.json()
        forecast_time = "2025-04-06 14:00:00"
        forecast = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
        
        return {
            "RainProbability": forecast["pop"] if forecast else 0,
            "Temperature": forecast["main"]["temp"] if forecast else 20,
            "Humidity": forecast["main"]["humidity"] if forecast else 60
        }
    except:
        return {"RainProbability": 0, "Temperature": 14.5, "Humidity": 60}

# ====================== FEATURE ENGINEERING ======================

def create_features(qualifying_df, historical_df, weather_data):
    """Combine all data sources into feature matrix"""
    wet_performance = {
        "VER": 0.92, "NOR": 0.95, "PIA": 0.94, "LEC": 0.93, "RUS": 0.91,
        "ANT": 0.89, "HAD": 0.88, "HAM": 0.96, "ALB": 0.90, "BEA": 0.87,
        "GAS": 0.93, "SAI": 0.94, "ALO": 0.97, "LAW": 0.89, "TSU": 0.91,
        "HUL": 0.92, "BOR": 0.85, "OCO": 0.93, "DOO": 0.86, "STR": 0.90
    }

    df = qualifying_df.merge(
        historical_df,
        left_on="DriverCode",
        right_on="Driver",
        how="left"
    )

    df["WetPerformance"] = df["DriverCode"].map(wet_performance)
    df["RainImpact"] = df["WetPerformance"] * weather_data["RainProbability"]

    df["SectorConsistency"] = (
        df[["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]].std(axis=1) /
        df[["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]].mean(axis=1)
    )

    df["Temperature"] = weather_data["Temperature"]
    df["Humidity"] = weather_data["Humidity"]

    # ✅ Explicitly retain Driver and Team from qualifying_df
    df["Driver"] = qualifying_df["Driver"]
    df["Team"] = qualifying_df["Team"]

    return df

# ====================== MODEL TRAINING ======================

def train_model(X, y):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae:.3f} seconds")
    
    return pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def analyze_team_strength(features_df):
    # Basic stats
    print("TeamStrength Summary Stats:")
    print(features_df["TeamStrength"].describe())

    # Skewness value
    skew_val = skew(features_df["TeamStrength"].dropna())
    print(f"\nSkewness of TeamStrength: {skew_val:.3f}")
    
    # Plot distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(features_df["TeamStrength"], kde=True, bins=10, color='skyblue')
    plt.title(f"TeamStrength Distribution (Skewness: {skew_val:.3f})")
    plt.xlabel("TeamStrength")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Call this after you create features_df


# ====================== MAIN EXECUTION ======================
def main():
    qualifying_df = load_qualifying_data()
    historical_df = load_historical_data()
    weather_data = get_weather_data()
    
    features_df = create_features(qualifying_df, historical_df, weather_data)
    
    feature_cols = [
        "QualifyingTime", "TeamStrength", "Sector1Time_s", 
        "Sector2Time_s", "Sector3Time_s", "WetPerformance",
        "RainImpact", "SectorConsistency", "Temperature", "Humidity"
    ]
    
    X = features_df[feature_cols]
    y = features_df["LapTime_s"]

    # Extract Driver and Team before filtering
    driver_team_df = features_df[["Driver", "Team"]].copy()

    # Drop rows with NaNs
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    driver_team_df = driver_team_df[valid_mask].reset_index(drop=True)
    features_df = features_df[valid_mask].reset_index(drop=True)

    model = train_model(X, y)
    
    features_df["PredictedRaceTime"] = model.predict(X)
    features_df["Driver"] = driver_team_df["Driver"]
    features_df["Team"] = driver_team_df["Team"]
    
    final_predictions = (
        features_df[["Driver", "Team", "QualifyingTime", "PredictedRaceTime"]]
        .sort_values("PredictedRaceTime")
        .reset_index(drop=True)
    )
    
    final_predictions["Position"] = range(1, len(final_predictions)+1)
    final_predictions["DeltaToLeader"] = (
        final_predictions["PredictedRaceTime"] - 
        final_predictions["PredictedRaceTime"].iloc[0]
    )
    
    print("\n2025 Japanese GP Predicted Results:")
    print(final_predictions.to_string(index=False))
    
    print(f"\nPredicted Winner: {final_predictions.iloc[0]['Driver']}")
    print(f"Predicted Pole-to-Win Conversion: {final_predictions.iloc[0]['Driver'] == qualifying_df.iloc[0]['Driver']}")
    analyze_team_strength(features_df)
        # ====================== ISSUE ANALYSIS ======================
    def analyze_prediction_issues(model, features_df, driver="Carlos Sainz"):
        print(f"\n🔍 Detailed Feature Comparison for {driver} and Top Rivals\n")

        # Get important columns
        feature_cols = [
            "QualifyingTime", "TeamStrength", "Sector1Time_s", 
            "Sector2Time_s", "Sector3Time_s", "WetPerformance",
            "RainImpact", "SectorConsistency", "Temperature", "Humidity"
        ]

        # Select driver and top 3 competitors
        rivals = ["Max Verstappen", "Lando Norris"]
        compare_drivers = rivals + [driver]
        comparison = features_df[features_df["Driver"].isin(compare_drivers)][
            ["Driver", "Team", "PredictedRaceTime"] + feature_cols
        ].sort_values("PredictedRaceTime")

        print(comparison.to_string(index=False))

        # Feature importances
        model_reg = model.named_steps["model"]
        importances = model_reg.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        print("\n📊 Feature Importances:")
        print(importance_df)

        # Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="mako")
        plt.title("Feature Importance in Race Time Prediction")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    # Call the function to inspect Carlos Sainz and rivals
    analyze_prediction_issues(model, features_df, driver="Carlos Sainz")


if __name__ == "__main__":
    main()
