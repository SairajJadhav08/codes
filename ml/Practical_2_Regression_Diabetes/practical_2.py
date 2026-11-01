"""
=====================================================================
 ML PRACTICAL 2
 Title   : Regression Analysis
 Aim     : Perform univariate analysis, bivariate analysis (linear
           and logistic regression) and multiple regression on the
           Pima Indians Diabetes dataset.
 Dataset : diabetes.csv (kept in this folder)
           Outcome = 1 means the person has diabetes, 0 means not
=====================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

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


heading("ML PRACTICAL 2 : REGRESSION ANALYSIS (DIABETES DATASET)")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

df = pd.read_csv(os.path.join(FOLDER, "diabetes.csv"))
print("Shape:", df.shape)
print(df.head())
print("\nSummary statistics:")
print(df.describe().round(2))

# ------------------------------------------------------------------
# STEP 2 : Univariate analysis (one column at a time)
# ------------------------------------------------------------------
heading("STEP 2 : Univariate analysis")

stats = pd.DataFrame({
    "Mean": df.mean(),
    "Median": df.median(),
    "Mode": df.mode().iloc[0],           # most frequent value
    "Variance": df.var(),
    "Std Dev": df.std(),
    "Skewness": df.skew(),               # > 0 means long right tail
    "Kurtosis": df.kurt(),               # > 0 means sharper peak than normal
})
print(stats.round(3))

print("\nFrequency (5 most common values) of each column:")
for column in df.columns:
    print(f"  {column:25s}: {df[column].value_counts().head(5).to_dict()}")

# ------------------------------------------------------------------
# STEP 3 : Bivariate analysis - Linear Regression
# ------------------------------------------------------------------
heading("STEP 3 : Linear Regression")

features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]                       # input columns
y = df["Outcome"]                      # target column

model_linear = LinearRegression()
model_linear.fit(X, y)
predictions_linear = model_linear.predict(X)

print("Linear Regression Coefficients:")
for feature, coef in zip(features, model_linear.coef_):
    print(f"  {feature:25s}: {coef:.6f}")
print(f"  {'Intercept':25s}: {model_linear.intercept_:.6f}")
print(f"\nR2 score : {r2_score(y, predictions_linear):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y, predictions_linear)):.4f}")

# ------------------------------------------------------------------
# STEP 4 : Bivariate analysis - Logistic Regression
# ------------------------------------------------------------------
heading("STEP 4 : Logistic Regression")

model_logistic = LogisticRegression(max_iter=1000)   # more iterations so it converges
model_logistic.fit(X, y)
predictions_logistic = model_logistic.predict(X)

print("Logistic Regression Coefficients:")
for feature, coef in zip(features, model_logistic.coef_[0]):
    print(f"  {feature:25s}: {coef:.6f}")
print(f"\nAccuracy on the data: {accuracy_score(y, predictions_logistic) * 100:.2f} %")

# ------------------------------------------------------------------
# STEP 5 : Multiple Regression (OLS) with statsmodels
# ------------------------------------------------------------------
heading("STEP 5 : Multiple Regression (Ordinary Least Squares)")

X_all = df.drop("Outcome", axis=1)     # all 8 independent variables
X_all = sm.add_constant(X_all)         # add the intercept column
results = sm.OLS(y, X_all).fit()
print(results.summary())

# ------------------------------------------------------------------
# STEP 6 : Correlation with the Outcome
# ------------------------------------------------------------------
heading("STEP 6 : Correlation of each column with Outcome")

corr = df.corr()
print(corr["Outcome"].drop("Outcome").sort_values(ascending=False).round(3))

# ------------------------------------------------------------------
# STEP 7 : Visualizations
# ------------------------------------------------------------------
heading("STEP 7 : Visualizations")

# 1. Correlation matrix
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns, rotation=90)
ax.set_yticklabels(df.columns)
ax.set_title("Correlation Matrix")
plt.tight_layout()

# 2. Scatter plot matrix (every column against every other column)
scatter_matrix(df, figsize=(14, 14))
plt.suptitle("Scatter Plot Matrix")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
