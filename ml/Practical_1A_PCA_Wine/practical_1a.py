"""
=====================================================================
 ML PRACTICAL 1A
 Title   : Feature Transformation using PCA
           (Principal Component Analysis)
 Aim     : Apply PCA on the Wine dataset to reduce its 13 features
           to a few components and separate the wine classes.
 Dataset : Wine.csv (kept in this folder)
           Customer_Segment = wine class (1, 2 or 3)
=====================================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 14)   # show all columns

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("ML PRACTICAL 1A : FEATURE TRANSFORMATION USING PCA (WINE DATASET)")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

df = pd.read_csv(os.path.join(FOLDER, "Wine.csv"))
print(df.head())
print("\nShape          :", df.shape)
print("Wine classes   :", sorted(df["Customer_Segment"].unique()))
print("\nMissing values in each column:")
print(df.isnull().sum())

# ------------------------------------------------------------------
# STEP 2 : Separate features and target, then standardize
# ------------------------------------------------------------------
heading("STEP 2 : Standardize the features")

X = df.drop("Customer_Segment", axis=1)    # 13 chemical features
y = df["Customer_Segment"]                 # wine class

# PCA needs all features on the same scale (mean 0, std 1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X_scaled.head().round(3))

# ------------------------------------------------------------------
# STEP 3 : Apply PCA with all components
# ------------------------------------------------------------------
heading("STEP 3 : Explained variance of each principal component")

pca = PCA()                                 # keep all 13 components
pca.fit(X_scaled)

variance = pd.DataFrame({
    "Component": [f"PC{i}" for i in range(1, len(X.columns) + 1)],
    "Variance %": (pca.explained_variance_ratio_ * 100).round(2),
    "Cumulative %": (pca.explained_variance_ratio_.cumsum() * 100).round(2),
})
print(variance.to_string(index=False))

# Smallest number of components that keep 95% of the information
n_95 = int((pca.explained_variance_ratio_.cumsum() < 0.95).sum() + 1)
print(f"\nComponents needed to keep 95% of the variance: {n_95}")

# ------------------------------------------------------------------
# STEP 4 : Reduce the dimensions
# ------------------------------------------------------------------
heading("STEP 4 : Reduce the dimensions")

pca_95 = PCA(n_components=n_95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print("Original data shape        :", X.shape)
print(f"Reduced data shape (95%)   : {X_pca_95.shape}")

pca_2 = PCA(n_components=2)                 # 2 components for a 2D graph
X_pca_2 = pca_2.fit_transform(X_scaled)
print("Reduced data shape (2D)    :", X_pca_2.shape)
print(f"Variance kept by 2 components: {pca_2.explained_variance_ratio_.sum() * 100:.2f} %")

print("\nFirst 5 rows after PCA (2 components):")
print(pd.DataFrame(X_pca_2, columns=["PC1", "PC2"]).head().round(3))

# ------------------------------------------------------------------
# STEP 5 : Visualizations
# ------------------------------------------------------------------
heading("STEP 5 : Visualizations")

# 1. Cumulative explained variance (helps choose the number of components)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 14), pca.explained_variance_ratio_.cumsum(), marker="o", linestyle="--")
plt.axhline(y=0.95, color="red", linestyle=":", label="95% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance Ratio")
plt.legend()
plt.grid(True)

# 2. Wine classes in 2D after PCA
plt.figure(figsize=(8, 6))
colors = {1: "red", 2: "blue", 3: "green"}
for wine_class, color in colors.items():
    points = X_pca_2[y == wine_class]
    plt.scatter(points[:, 0], points[:, 1], c=color, label=f"Wine class {wine_class}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Wine Classes in 2 Dimensions")
plt.legend()
plt.grid(True)

print("Opening all graphs... close the graph windows to finish.")
plt.show()
