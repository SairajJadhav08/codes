"""
=====================================================================
 DMV PRACTICAL 3
 Title   : Data Cleaning and Preparation
 Problem : Analyzing Customer Churn in a Telecommunications Company
 Aim     : Clean and prepare the telecom customer data and find the
           factors that make customers leave (churn).
 Dataset : Telecom_Customer_Churn.csv (kept in this folder)
=====================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 12)   # limit columns shown in the terminal

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("DMV PRACTICAL 3 : CUSTOMER CHURN - DATA CLEANING AND PREPARATION")

# ------------------------------------------------------------------
# STEP 1 : Load the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load the dataset")

data = pd.read_csv(os.path.join(FOLDER, "Telecom_Customer_Churn.csv"))
print("Index   :", data.index)
print("Shape   :", data.shape)
print("Columns :", list(data.columns))

# ------------------------------------------------------------------
# STEP 2 : Explore the data
# ------------------------------------------------------------------
heading("STEP 2 : Explore the data")

print("First 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())
print("\nUnique values in each column:")
print(data.nunique())
print()
data.info()

# ------------------------------------------------------------------
# STEP 3 : Handle missing values
# ------------------------------------------------------------------
heading("STEP 3 : Handle missing values")

print("Missing values (as loaded):", data.isnull().sum().sum())

# TotalCharges is stored as text and has blank spaces " " for new customers
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")  # " " -> NaN
print("Blank TotalCharges found :", data["TotalCharges"].isnull().sum())

# These customers have tenure 0, so they have not been charged yet -> 0
data["TotalCharges"] = data["TotalCharges"].fillna(0)
print("Missing values after fix :", data.isnull().sum().sum())

# ------------------------------------------------------------------
# STEP 4 : Remove duplicate records
# ------------------------------------------------------------------
heading("STEP 4 : Remove duplicate records")

print("Rows before removing duplicates:", len(data))
data = data.drop_duplicates()
print("Rows after removing duplicates :", len(data))

# ------------------------------------------------------------------
# STEP 5 : Fix inconsistent values
# ------------------------------------------------------------------
heading("STEP 5 : Standardize inconsistent values")

print("Values in OnlineSecurity before:", list(data["OnlineSecurity"].unique()))

# "No internet service" / "No phone service" simply mean "No"
service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
data[service_cols] = data[service_cols].replace({"No internet service": "No",
                                                 "No phone service": "No"})
print("Values in OnlineSecurity after :", list(data["OnlineSecurity"].unique()))

# ------------------------------------------------------------------
# STEP 6 : Statistics and frequency distribution
# ------------------------------------------------------------------
heading("STEP 6 : Statistics and frequency distribution")

print(data.describe())

unique, counts = np.unique(data["tenure"], return_counts=True)   # how often each tenure occurs
print("\nTenure (months) frequency - first 15 values:")
print(pd.DataFrame({"tenure": unique, "customers": counts}).head(15).to_string(index=False))

print("\nCustomers by contract type:")
print(data["Contract"].value_counts())

# ------------------------------------------------------------------
# STEP 7 : Detect outliers with the IQR rule
# ------------------------------------------------------------------
heading("STEP 7 : Detect outliers (IQR method)")

for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1                                         # inter-quartile range
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr            # outlier limits
    outliers = data[(data[col] < low) | (data[col] > high)]
    print(f"{col:15s}: limits {low:9.2f} to {high:9.2f} -> outliers = {len(outliers)}")

# ------------------------------------------------------------------
# STEP 8 : Insights - which customers churn?
# ------------------------------------------------------------------
heading("STEP 8 : Churn insights")

data["Churn_Flag"] = (data["Churn"] == "Yes").astype(int)        # Yes -> 1, No -> 0
print(f"Overall churn rate: {data['Churn_Flag'].mean() * 100:.2f} %")

for col in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]:
    rate = (data.groupby(col)["Churn_Flag"].mean() * 100).round(2)
    print(f"\nChurn rate (%) by {col}:")
    print(rate.sort_values(ascending=False))

# ------------------------------------------------------------------
# STEP 9 : Prepare data for machine learning
# ------------------------------------------------------------------
heading("STEP 9 : Prepare data for modeling (encode + split)")

X = data.drop(["customerID", "Churn", "Churn_Flag"], axis=1)    # input features
y = data["Churn_Flag"]                                          # target (1 = churned)
X = pd.get_dummies(X, drop_first=True)                          # text -> 0/1 columns
print("Features after encoding:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape:", y_test.shape)

# Save the cleaned data for later use
data.drop(columns="Churn_Flag").to_csv(
    os.path.join(FOLDER, "Telecom_Customer_Churn_cleaned.csv"), index=False)
print("\nCleaned data saved as Telecom_Customer_Churn_cleaned.csv")

# ------------------------------------------------------------------
# STEP 10 : Visualizations
# ------------------------------------------------------------------
heading("STEP 10 : Visualizations")

# 1. Box plots of the numeric columns
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    ax.boxplot(data[col])
    ax.set_title("Boxplot of " + col)
plt.tight_layout()

# 2. Pair plot of numeric columns, coloured by churn
sns.pairplot(data[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]], hue="Churn")

# 3. How many customers churned
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=data)
plt.title("Number of Customers Who Churned")

# 4. Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", hue="Churn", data=data)
plt.title("Churn by Contract Type")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
