"""
=====================================================================
 ML PRACTICAL 1B
 Title   : Feature Transformation using LDA
           (Linear Discriminant Analysis)
 Aim     : Reduce the Iris features with LDA, train a classifier on
           the reduced data and predict the species of a new flower.
 Dataset : Iris.csv (kept in this folder)
=====================================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("ML PRACTICAL 1B : FEATURE TRANSFORMATION USING LDA (IRIS DATASET)")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

df = pd.read_csv(os.path.join(FOLDER, "Iris.csv"))
print(df)
print("\nSpecies:", list(df["Species"].unique()))

# ------------------------------------------------------------------
# STEP 2 : Separate features and target, then scale
# ------------------------------------------------------------------
heading("STEP 2 : Scale the features")

X = df.drop(["Id", "Species"], axis=1)     # 4 flower measurements
y = df["Species"]                          # flower species

scaler = StandardScaler()                  # mean 0, std 1 for every feature
X_scaled = scaler.fit_transform(X)
print("Shape before LDA:", X_scaled.shape)

# ------------------------------------------------------------------
# STEP 3 : Apply LDA (4 features -> 2 features)
# ------------------------------------------------------------------
heading("STEP 3 : Apply LDA")

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)     # LDA uses the labels to separate classes
print("Shape after LDA :", X_lda.shape)
print("Variance explained by each LDA component:", (lda.explained_variance_ratio_ * 100).round(2), "%")

# ------------------------------------------------------------------
# STEP 4 : Train a classifier on the LDA features
# ------------------------------------------------------------------
heading("STEP 4 : Train Logistic Regression on LDA features")

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print("Model trained successfully.")

# ------------------------------------------------------------------
# STEP 5 : Evaluate the model
# ------------------------------------------------------------------
heading("STEP 5 : Evaluate the model")

y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# STEP 6 : Predict the species of a new flower
# ------------------------------------------------------------------
heading("STEP 6 : Predict a new flower")

# Sepal length, sepal width, petal length, petal width (in cm)
new_flower = pd.DataFrame([[6.7, 3.0, 5.2, 2.3]], columns=X.columns)
print("New flower measurements:")
print(new_flower.to_string(index=False))

new_flower_scaled = scaler.transform(new_flower)     # same scaling as training data
new_flower_lda = lda.transform(new_flower_scaled)    # same LDA as training data
predicted_species = classifier.predict(new_flower_lda)[0]

species_number = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
print(f"\nPredicted species: {predicted_species} (class {species_number[predicted_species]})")

# ------------------------------------------------------------------
# STEP 7 : Visualization
# ------------------------------------------------------------------
heading("STEP 7 : Visualization")

plt.figure(figsize=(8, 6))
for species, color in zip(species_number, ["red", "green", "blue"]):
    points = X_lda[y == species]
    plt.scatter(points[:, 0], points[:, 1], c=color, label=species)
plt.scatter(new_flower_lda[0, 0], new_flower_lda[0, 1], c="black", marker="*",
            s=300, label="New flower")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("LDA: Iris Species in 2 Dimensions")
plt.legend()
plt.grid(True)

print("Opening the graph... close the graph window to finish.")
plt.show()
