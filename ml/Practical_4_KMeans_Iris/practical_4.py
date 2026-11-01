"""
=====================================================================
 ML PRACTICAL 4
 Title   : Implementation of K-Means Clustering
 Aim     : Apply K-Means clustering on the Iris dataset and find the
           best number of clusters using the Elbow method.
 Dataset : Iris.csv (kept in this folder)
=====================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("ML PRACTICAL 4 : K-MEANS CLUSTERING ON IRIS DATASET")

# ------------------------------------------------------------------
# STEP 1 : Load the data
# ------------------------------------------------------------------
heading("STEP 1 : Load the data")

data = pd.read_csv(os.path.join(FOLDER, "Iris.csv"))
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# ------------------------------------------------------------------
# STEP 2 : View and describe the data
# ------------------------------------------------------------------
heading("STEP 2 : View and describe the data")

print("Number of rows:", len(data))
print("Shape         :", data.shape)
for i, col in enumerate(data.columns):
    print(f"Column number {1 + i} is {col}")
print("\nData types:")
print(data.dtypes)
print()
data.info()
print("\nSummary statistics:")
print(data.describe())

# ------------------------------------------------------------------
# STEP 3 : Check and clean the data
# ------------------------------------------------------------------
heading("STEP 3 : Check and clean the data")

print("Missing values in each column:")
print(data.isna().sum())
data = data.drop("Id", axis=1)             # Id is just a row number, not a feature
print("\nAfter dropping Id column:")
print(data.head())
print("\nCount of each species:")
print(data["Species"].value_counts())

# ------------------------------------------------------------------
# STEP 4 : Separate the features used for clustering
# ------------------------------------------------------------------
heading("STEP 4 : Select features for clustering")

target_data = data.iloc[:, 4]              # Species (only used to check results)
clustering_data = data.iloc[:, [0, 1, 2, 3]]   # 4 measurements used for clustering
print(clustering_data.head())

# ------------------------------------------------------------------
# STEP 5 : Elbow method to find the best number of clusters
# ------------------------------------------------------------------
heading("STEP 5 : Elbow method")

# WCSS = Within-Cluster Sum of Squares (smaller = tighter clusters)
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(clustering_data)
    wcss.append(km.inertia_)
    print(f"k = {k:2d}  ->  WCSS = {km.inertia_:.2f}")

print("\nThe curve bends sharply (the 'elbow') at k = 3, so we use 3 clusters.")

# ------------------------------------------------------------------
# STEP 6 : Apply K-Means with 3 clusters
# ------------------------------------------------------------------
heading("STEP 6 : K-Means clustering with k = 3")

kms = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
clusters = clustering_data.copy()
clusters["Cluster_Prediction"] = kms.fit_predict(clustering_data)
print(clusters.head())

print("\nCluster centres (centroids):")
print(pd.DataFrame(kms.cluster_centers_, columns=clustering_data.columns).round(3))

print("\nNumber of flowers in each cluster:")
print(clusters["Cluster_Prediction"].value_counts().sort_index())

# Compare clusters with the real species (not used during clustering)
print("\nClusters vs real species:")
print(pd.crosstab(clusters["Cluster_Prediction"], target_data))

# ------------------------------------------------------------------
# STEP 7 : Visualizations
# ------------------------------------------------------------------
heading("STEP 7 : Visualizations")

# 1. Raw data before clustering
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data["SepalLengthCm"], y=data["SepalWidthCm"], s=70, color="#f73434")
plt.xlabel("Sepal Length (in cm)")
plt.ylabel("Sepal Width (in cm)")
plt.title("Sepal Length vs Width", fontsize=16)

# 2. Elbow graph
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.axvline(x=3, ls="--")
plt.xlabel("No. of Clusters (k)")
plt.ylabel("WCSS")
plt.title("The Elbow Method", fontsize=16)
plt.grid(True)

# 3. Final clusters with their centroids
plt.figure(figsize=(12, 6))
colors = ["teal", "lime", "magenta"]
for cluster_id in range(3):
    points = clusters[clusters["Cluster_Prediction"] == cluster_id]
    name = target_data[points.index].mode()[0]       # most common species in this cluster
    plt.scatter(points["SepalLengthCm"], points["SepalWidthCm"], s=70,
                c=colors[cluster_id], label=f"Cluster {cluster_id} ({name})")
plt.scatter(kms.cluster_centers_[:, 0], kms.cluster_centers_[:, 1], s=170,
            c="yellow", edgecolor="black", label="Centroids")
plt.legend(loc="upper right")
plt.xlabel("Sepal Length (in cm)")
plt.ylabel("Sepal Width (in cm)")
plt.title("Clusters", fontsize=16)

print("Opening all graphs... close the graph windows to finish.")
plt.show()
