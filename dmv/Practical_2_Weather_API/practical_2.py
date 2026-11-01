"""
=====================================================================
 DMV PRACTICAL 2
 Title   : Analyzing Weather Data from OpenWeatherMap API
 Aim     : Fetch weather data for a location from the OpenWeatherMap
           API, clean it, analyze it and visualize weather patterns.
 Dataset : Live data from the API (5-day forecast, every 3 hours).
           A copy is saved as weather_backup.json for offline use.
=====================================================================
"""

import os
import sys
import json
import warnings

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")   # allow special letters in city names
warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 10)   # show all columns

# Folder of this script, used for the backup file
FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FILE = os.path.join(FOLDER, "weather_backup.json")

# API settings (get your own free key at https://openweathermap.org/api)
API_KEY = "fb365aa6104829b44455572365ff3b4e"
LAT, LON = 18.184135, 74.610764            # location used in the lab notebook
API_URL = (f"https://api.openweathermap.org/data/2.5/forecast"
           f"?lat={LAT}&lon={LON}&appid={API_KEY}")


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("DMV PRACTICAL 2 : ANALYZING WEATHER DATA FROM OPENWEATHERMAP API")

# ------------------------------------------------------------------
# STEP 1 : Get weather data from the API
# ------------------------------------------------------------------
heading("STEP 1 : Fetch weather data from the API")

try:
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()                 # stop here if the API gave an error
    weather_data = response.json()              # convert reply to a Python dict
    with open(BACKUP_FILE, "w") as file:        # save a copy for offline use
        json.dump(weather_data, file)
    print("Live data received from the API (status code", response.status_code, ")")
except Exception as error:
    print("API request failed:", error)
    print("Using the saved backup file instead.")
    with open(BACKUP_FILE) as file:
        weather_data = json.load(file)

# ------------------------------------------------------------------
# STEP 2 : Explore the API response
# ------------------------------------------------------------------
heading("STEP 2 : Explore the API response")

city = weather_data["city"]
print("Keys in the response :", list(weather_data.keys()))
print("Location             :", city["name"], ",", city["country"])
print("Number of records    :", len(weather_data["list"]), "(one every 3 hours)")
print("First weather report :", weather_data["list"][0]["weather"][0]["description"])

# ------------------------------------------------------------------
# STEP 3 : Extract the useful weather attributes into a DataFrame
# ------------------------------------------------------------------
heading("STEP 3 : Extract weather attributes into a table")

records = weather_data["list"]
utc_offset = city["timezone"]                   # seconds ahead of UTC (India = 19800)

weather_df = pd.DataFrame({
    "Timestamp":   [pd.to_datetime(item["dt"] + utc_offset, unit="s") for item in records],
    "Temperature": [item["main"]["temp"] for item in records],               # Kelvin
    "Humidity":    [item["main"]["humidity"] for item in records],           # %
    "Wind_Speed":  [item["wind"]["speed"] for item in records],              # m/s
    "Rain_mm":     [item.get("rain", {}).get("3h", 0) for item in records],  # rain in 3 hours
    "Description": [item["weather"][0]["description"] for item in records],
})
weather_df.set_index("Timestamp", inplace=True)   # time becomes the row index
print(weather_df.head(10))

# ------------------------------------------------------------------
# STEP 4 : Clean the data
# ------------------------------------------------------------------
heading("STEP 4 : Clean the data")

print("Missing values in each column:")
print(weather_df.isnull().sum())
weather_df.fillna(0, inplace=True)                                    # fill any gaps with 0
weather_df = weather_df[~weather_df.index.duplicated()]              # drop repeated times
weather_df["Temperature"] = (weather_df["Temperature"] - 273.15).round(2)  # Kelvin -> Celsius
print("\nTemperature converted from Kelvin to Celsius:")
print(weather_df.head())

# ------------------------------------------------------------------
# STEP 5 : Data modeling (statistics and extreme values)
# ------------------------------------------------------------------
heading("STEP 5 : Weather statistics")

temp = weather_df["Temperature"]
print(f"Maximum temperature : {temp.max()} C at {temp.idxmax()}")
print(f"Minimum temperature : {temp.min()} C at {temp.idxmin()}")
print(f"Average temperature : {temp.mean():.2f} C")
print(f"Average humidity    : {weather_df['Humidity'].mean():.2f} %")
print(f"Average wind speed  : {weather_df['Wind_Speed'].mean():.2f} m/s")
print(f"Total rainfall      : {weather_df['Rain_mm'].sum():.2f} mm")

print("\nHow often each weather type appears:")
print(weather_df["Description"].value_counts())

# ------------------------------------------------------------------
# STEP 6 : Aggregate the data day by day
# ------------------------------------------------------------------
heading("STEP 6 : Daily summary (aggregation)")

daily = weather_df.resample("D").agg({
    "Temperature": "mean",      # average temperature of the day
    "Humidity": "mean",         # average humidity of the day
    "Wind_Speed": "mean",       # average wind speed of the day
    "Rain_mm": "sum",           # total rain of the day
}).round(2)
daily.index = daily.index.date  # show only the date
print(daily)

# ------------------------------------------------------------------
# STEP 7 : Correlation analysis
# ------------------------------------------------------------------
heading("STEP 7 : Correlation between weather attributes")

numeric_cols = ["Temperature", "Humidity", "Wind_Speed", "Rain_mm"]
correlation = weather_df[numeric_cols].corr().round(2)
print(correlation)

# ------------------------------------------------------------------
# STEP 8 : Visualizations
# ------------------------------------------------------------------
heading("STEP 8 : Visualizations")

# 1. Temperature every 3 hours
plt.figure(figsize=(12, 5))
weather_df["Temperature"].plot(color="red", marker=".")
plt.title("Temperature Over Time - " + city["name"])
plt.xlabel("Date and Time")
plt.ylabel("Temperature (C)")
plt.grid(True)

# 2. Mean daily temperature
plt.figure(figsize=(10, 6))
daily["Temperature"].plot(color="red", linestyle="-", marker="o")
plt.title("Mean Daily Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (C)")
plt.grid(True)

# 3. Mean daily humidity
plt.figure(figsize=(10, 6))
daily["Humidity"].plot(kind="bar", color="blue")
plt.title("Mean Daily Humidity")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Daily rainfall (precipitation)
plt.figure(figsize=(10, 6))
daily["Rain_mm"].plot(kind="bar", color="teal")
plt.title("Total Daily Rainfall")
plt.xlabel("Date")
plt.ylabel("Rain (mm)")
plt.xticks(rotation=45)
plt.tight_layout()

# 5. Temperature vs wind speed
plt.figure(figsize=(10, 6))
plt.scatter(weather_df["Temperature"], weather_df["Wind_Speed"], color="green")
plt.title("Temperature vs Wind Speed")
plt.xlabel("Temperature (C)")
plt.ylabel("Wind Speed (m/s)")
plt.grid(True)

# 6. Temperature vs humidity
plt.figure(figsize=(10, 6))
plt.scatter(weather_df["Temperature"], weather_df["Humidity"], color="purple")
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (C)")
plt.ylabel("Humidity (%)")
plt.grid(True)

# 7. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Weather Attributes")
plt.tight_layout()

print("Opening all graphs... close the graph windows to finish.")
plt.show()
