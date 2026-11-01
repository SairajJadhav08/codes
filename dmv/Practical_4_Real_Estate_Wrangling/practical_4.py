"""
=====================================================================
 DMV PRACTICAL 4
 Title   : Data Wrangling on Real Estate Market
 Aim     : Clean, transform and prepare Bengaluru house price data
           to understand the factors that influence house prices.
 Dataset : Bengaluru_House_Data.csv (kept in this folder)
           price is in Lakhs (1 Lakh = 1,00,000 Rupees)
=====================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 10)   # show all columns

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


def is_float(x):
    """Return True if the value can be converted to a number."""
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def convert_sqft_to_num(x):
    """Convert '2100 - 2850' to its average (2475.0) and '1056' to 1056.0.
    Values with units like '34.46Sq. Meter' cannot be converted -> None."""
    tokens = x.split("-")
    if len(tokens) == 2:                       # a range like "2100 - 2850"
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)                        # a normal number
    except ValueError:
        return None


heading("DMV PRACTICAL 4 : DATA WRANGLING ON REAL ESTATE MARKET")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

df1 = pd.read_csv(os.path.join(FOLDER, "Bengaluru_House_Data.csv"))
df1.columns = df1.columns.str.strip().str.lower()     # clean column names
print(df1.head())
print("\nShape   :", df1.shape)
print("Columns :", list(df1.columns))

df1["area_type"] = df1["area_type"].str.split().str.join(" ")   # fix double spaces
print("\nArea types:", list(df1["area_type"].unique()))
print("\nCount of each area type:")
print(df1["area_type"].value_counts())

# ------------------------------------------------------------------
# STEP 2 : Drop columns that are not needed
# ------------------------------------------------------------------
heading("STEP 2 : Drop unneeded columns")

df2 = df1.drop(["area_type", "society", "balcony", "availability"], axis="columns")
print("Shape after dropping columns:", df2.shape)

# ------------------------------------------------------------------
# STEP 3 : Handle missing values
# ------------------------------------------------------------------
heading("STEP 3 : Handle missing values")

print("Missing values in each column:")
print(df2.isnull().sum())
df3 = df2.dropna().copy()                     # very few rows are missing, so drop them
print("\nShape after dropping missing rows:", df3.shape)

# ------------------------------------------------------------------
# STEP 4 : Transform 'size' into a number of bedrooms (bhk)
# ------------------------------------------------------------------
heading("STEP 4 : Create 'bhk' column from 'size'")

print("Size values:", list(df3["size"].unique()[:12]), "...")
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))   # "2 BHK" -> 2
print(df3.head())
print("\nHouses with more than 20 bedrooms (suspicious):")
print(df3[df3.bhk > 20])

# ------------------------------------------------------------------
# STEP 5 : Standardize 'total_sqft' into numbers
# ------------------------------------------------------------------
heading("STEP 5 : Convert total_sqft to numbers")

print("Rows where total_sqft is not a plain number:")
print(df3[~df3["total_sqft"].apply(is_float)].head(10))

print("\nExample: convert_sqft_to_num('2100 - 2850') =", convert_sqft_to_num("2100 - 2850"))
print("Example: convert_sqft_to_num('34.46Sq. Meter') =", convert_sqft_to_num("34.46Sq. Meter"))

df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
df4 = df4[df4["total_sqft"].notnull()]        # drop values we could not convert
print("\nShape after conversion:", df4.shape)
print("Row 30 after conversion:")
print(df4.loc[30])

# ------------------------------------------------------------------
# STEP 6 : Feature engineering - price per square foot
# ------------------------------------------------------------------
heading("STEP 6 : Feature engineering (price_per_sqft)")

df5 = df4.copy()
df5["price_per_sqft"] = df5["price"] * 100000 / df5["total_sqft"]   # Lakhs -> Rupees
print(df5.head())
print("\nprice_per_sqft statistics:")
print(df5["price_per_sqft"].describe())

# ------------------------------------------------------------------
# STEP 7 : Reduce the number of locations
# ------------------------------------------------------------------
heading("STEP 7 : Group rare locations into 'other'")

df5["location"] = df5["location"].apply(lambda x: x.strip())   # remove extra spaces
location_stats = df5["location"].value_counts(ascending=False)
print("Total locations                 :", len(location_stats))
print("Locations with more than 10 rows:", len(location_stats[location_stats > 10]))
print("Locations with 10 or fewer rows :", len(location_stats[location_stats <= 10]))

rare_locations = location_stats[location_stats <= 10]
df5["location"] = df5["location"].apply(lambda x: "other" if x in rare_locations else x)
print("Locations after grouping        :", df5["location"].nunique())
print(df5.head(10))

# ------------------------------------------------------------------
# STEP 8 : Remove unrealistic rows (less than 300 sqft per bedroom)
# ------------------------------------------------------------------
heading("STEP 8 : Remove unrealistic houses")

print("Examples with less than 300 sqft per bedroom:")
print(df5[df5.total_sqft / df5.bhk < 300].head())
df6 = df5[~(df5.total_sqft / df5.bhk < 300)].copy()
print("\nShape before:", df5.shape, " after:", df6.shape)

# ------------------------------------------------------------------
# STEP 9 : Remove outliers using the IQR method
# ------------------------------------------------------------------
heading("STEP 9 : Remove outliers (IQR method)")

outlier_cols = ["total_sqft", "bath", "price", "bhk", "price_per_sqft"]
before_outliers = df6.copy()                  # keep a copy for the 'before' box plots

for col in outlier_cols:
    q1 = np.percentile(df6[col], 25)          # 25th percentile
    q3 = np.percentile(df6[col], 75)          # 75th percentile
    iqr = q3 - q1                             # inter-quartile range
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows_before = len(df6)
    df6 = df6[(df6[col] >= lower) & (df6[col] <= upper)]   # keep only normal values
    print(f"{col:15s}: removed {rows_before - len(df6):5d} outlier rows")

print("\nShape after removing outliers:", df6.shape)

# ------------------------------------------------------------------
# STEP 10 : Aggregate - average price by location and bhk
# ------------------------------------------------------------------
heading("STEP 10 : Aggregation (summary by location and bhk)")

top_locations = (df6[df6.location != "other"]
                 .groupby("location")["price_per_sqft"].mean()
                 .sort_values(ascending=False).head(10).round(2))
print("Top 10 most expensive locations (average price per sqft):")
print(top_locations)

print("\nAverage price (Lakhs) by number of bedrooms:")
print(df6.groupby("bhk")["price"].mean().round(2))

# ------------------------------------------------------------------
# STEP 11 : Encode categories and split into train / test sets
# ------------------------------------------------------------------
heading("STEP 11 : Encode 'location' and split the data")

X = df6.drop(["price", "size"], axis="columns")            # input features
X = pd.get_dummies(X, columns=["location"], drop_first=True)   # location -> 0/1 columns
y = df6["price"]                                            # target: house price
print("X shape after encoding:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape:", y_test.shape)

# Save the cleaned data for later use
df6.to_csv(os.path.join(FOLDER, "bhp_cleaned.csv"), index=False)
print("\nCleaned data saved as bhp_cleaned.csv")

# ------------------------------------------------------------------
# STEP 12 : Visualizations
# ------------------------------------------------------------------
heading("STEP 12 : Visualizations")

# 1. Box plots before removing outliers
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(before_outliers[col])
    ax.set_title(col)
fig.suptitle("Box Plots BEFORE Removing Outliers")
plt.tight_layout()

# 2. Box plots after removing outliers
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(df6[col])
    ax.set_title(col)
fig.suptitle("Box Plots AFTER Removing Outliers")
plt.tight_layout()

# 3. Price per sqft distribution
plt.figure(figsize=(8, 5))
plt.hist(df6["price_per_sqft"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of Price per Square Foot")
plt.xlabel("Price per sqft (Rupees)")
plt.ylabel("Number of Houses")

# 4. Top 10 expensive locations
plt.figure(figsize=(10, 5))
top_locations.plot(kind="bar", color="orange")
plt.title("Top 10 Locations by Average Price per Sqft")
plt.xlabel("Location")
plt.ylabel("Price per sqft (Rupees)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# 5. Area vs price
plt.figure(figsize=(8, 5))
plt.scatter(df6["total_sqft"], df6["price"], s=5, alpha=0.5)
plt.title("Total Sqft vs Price")
plt.xlabel("Total Sqft")
plt.ylabel("Price (Lakhs)")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
