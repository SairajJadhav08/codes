"""
=====================================================================
 DMV PRACTICAL 5
 Title   : Data Visualization using Matplotlib
 Problem : Analyzing Air Quality Index (AQI) Trends in a City
 Aim     : Visualize the trends and patterns of air pollutants
           (SO2, NO2, RSPM, SPM, PM2.5) across Indian states and cities.
 Dataset : air_quality_data.csv (India air quality data, kept in this folder)
           Pollutant values are in micrograms per cubic metre (ug/m3)
=====================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 13)   # show all columns

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))

VALUE_COLS = ["so2", "no2", "rspm", "spm", "pm2_5"]   # pollutant columns used everywhere
CITY = "Delhi"                                          # city used for the trend graph


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("DMV PRACTICAL 5 : ANALYZING AIR QUALITY INDEX (AQI) TRENDS")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

print("Loading the dataset (4.3 lakh rows, may take a few seconds)...")
data = pd.read_csv(os.path.join(FOLDER, "air_quality_data.csv"),
                   encoding="cp1252", low_memory=False)
print(data.head())
print("\nShape:", data.shape)
print()
data.info()

# ------------------------------------------------------------------
# STEP 2 : Clean the data
# ------------------------------------------------------------------
heading("STEP 2 : Clean state names and area types")

# Old state names -> new state names
data["state"] = data["state"].replace({"Uttaranchal": "Uttarakhand"})
data.loc[data["location"] == "Jamshedpur", "state"] = "Jharkhand"

# Many spellings of the same area type -> one short code
types = {
    "Residential": "R",
    "Residential and others": "RO",
    "Residential, Rural and other Areas": "RRO",
    "Industrial Area": "I",
    "Industrial Areas": "I",
    "Industrial": "I",
    "Sensitive Area": "S",
    "Sensitive Areas": "S",
    "Sensitive": "S",
}
data["type"] = data["type"].replace(types).fillna("RRO")    # missing type -> RRO
print("Area types after cleaning:")
print(data["type"].value_counts())

# Convert the date column from text to real dates
data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="coerce")
data["year"] = data["date"].dt.year.astype("Int64")         # year as a whole number
print("\nData covers years", data["year"].min(), "to", data["year"].max())

# Keep the real city readings BEFORE filling gaps, so trends stay honest
city_data = data[data["location"] == CITY].copy()
city_trend = city_data.groupby("year")[["so2", "no2", "rspm"]].mean()

# ------------------------------------------------------------------
# STEP 3 : Fill missing pollutant values
# ------------------------------------------------------------------
heading("STEP 3 : Fill missing pollutant values with the mean")

print("Missing pollutant values BEFORE:")
print(data[VALUE_COLS].isnull().sum())

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")   # replace NaN by column mean
data[VALUE_COLS] = imputer.fit_transform(data[VALUE_COLS])

print("\nMissing pollutant values AFTER:")
print(data[VALUE_COLS].isnull().sum())

# ------------------------------------------------------------------
# STEP 4 : Summary statistics
# ------------------------------------------------------------------
heading("STEP 4 : Summary statistics of pollutants")

print(data[VALUE_COLS].describe().round(2))

print("\nAverage pollutant level by area type:")
print(data.groupby("type")[VALUE_COLS].mean().round(2))

print(f"\nYearly average pollutant levels in {CITY} (real readings, NaN = not measured):")
print(city_trend.round(2))

# ------------------------------------------------------------------
# STEP 5 : Visualization functions
# ------------------------------------------------------------------


def top_and_bottom_10_states(indicator="so2"):
    """Bar charts of the 10 most and 10 least polluted states."""
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    ind = (data[[indicator, "state"]].groupby("state", as_index=False).median()
           .sort_values(by=indicator, ascending=False))
    sns.barplot(x="state", y=indicator, data=ind[:10], ax=ax[0], color="red")
    ax[0].set_title(f"Top 10 states by {indicator} (median)")
    ax[0].set_ylabel(f"{indicator} (ug/m3)")
    sns.barplot(x="state", y=indicator, data=ind[-10:], ax=ax[1], color="green")
    ax[1].set_title(f"Bottom 10 states by {indicator} (median)")
    ax[1].set_ylabel(f"{indicator} (ug/m3)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)     # tilt state names so they do not overlap
    plt.tight_layout()


def highest_levels_recorded(indicator="so2"):
    """Bar chart of the highest value ever recorded in each state."""
    plt.figure(figsize=(16, 8))
    ind = data[[indicator, "state"]].groupby("state", as_index=False).max()
    sns.barplot(x="state", y=indicator, data=ind)
    plt.title(f"Highest ever {indicator} levels recorded by state")
    plt.ylabel(f"{indicator} (ug/m3)")
    plt.xticks(rotation=90)
    plt.tight_layout()


def type_avg():
    """Bar chart of average pollutant levels for each area type."""
    type_means = data[VALUE_COLS + ["type"]].groupby("type").mean()
    type_means.plot(kind="bar", figsize=(10, 6))
    plt.xticks(rotation=0)
    plt.title("Pollutant average by area type")
    plt.ylabel("Level (ug/m3)")


def location_avgs(state, indicator="so2"):
    """Bar chart of average pollutant level for each city in a state."""
    plt.figure(figsize=(10, 6))
    locs = data[data["state"] == state].groupby("location", as_index=False)[indicator].mean()
    sns.barplot(x="location", y=indicator, data=locs)
    plt.title(f"Location-wise average {indicator} in {state}")
    plt.ylabel(f"{indicator} (ug/m3)")
    plt.xticks(rotation=90)
    plt.tight_layout()


# ------------------------------------------------------------------
# STEP 6 : Draw the graphs
# ------------------------------------------------------------------
heading("STEP 6 : Visualizations")

# 1-2. Most and least polluted states
top_and_bottom_10_states("so2")
top_and_bottom_10_states("no2")

# 3-4. Highest levels ever recorded
highest_levels_recorded("no2")
highest_levels_recorded("rspm")

# 5. Pollutant average by area type
type_avg()

# 6. City-wise average in one state
location_avgs("Bihar", "no2")

# 7. Pollutant trend over the years in one city (line plot)
plt.figure(figsize=(12, 6))
for col in ["so2", "no2", "rspm"]:
    plt.plot(city_trend.index, city_trend[col], marker="o", label=col.upper())
plt.title(f"Yearly Pollutant Trend in {CITY}")
plt.xlabel("Year")
plt.ylabel("Average level (ug/m3)")
plt.legend()
plt.grid(True)

# 8. Distribution of pollutant values (box plot)
plt.figure(figsize=(10, 6))
plt.boxplot([data["so2"], data["no2"], data["rspm"]], showfliers=False)
plt.xticks([1, 2, 3], ["SO2", "NO2", "RSPM"])
plt.title("Distribution of Pollutant Levels (outliers hidden)")
plt.ylabel("Level (ug/m3)")

# 9. Relationship between NO2 and RSPM (real readings of one city)
sample = city_data.dropna(subset=["no2", "rspm"])
plt.figure(figsize=(8, 6))
plt.scatter(sample["no2"], sample["rspm"], s=8, alpha=0.4, color="purple")
plt.title(f"NO2 vs RSPM in {CITY}")
plt.xlabel("NO2 (ug/m3)")
plt.ylabel("RSPM (ug/m3)")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
