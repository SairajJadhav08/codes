"""
=====================================================================
 ML PRACTICAL 5
 Title   : Implement Random Forest Classifier
 Aim     : Build a Random Forest model to predict the safety
           (acceptability class) of a car.
 Dataset : car_evaluation.csv (UCI Car Evaluation, kept in this folder)
           The file has no header row, so column names are added below.
=====================================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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


heading("ML PRACTICAL 5 : RANDOM FOREST CLASSIFIER - CAR SAFETY PREDICTION")

# ------------------------------------------------------------------
# STEP 1 : Load the dataset and add column names
# ------------------------------------------------------------------
heading("STEP 1 : Load the dataset")

columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(os.path.join(FOLDER, "car_evaluation.csv"), header=None, names=columns)
print(df.head())

# ------------------------------------------------------------------
# STEP 2 : Dataset information
# ------------------------------------------------------------------
heading("STEP 2 : Dataset information")

print("Dataset Shape:", df.shape)
print()
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------------
# STEP 3 : Check the target classes
# ------------------------------------------------------------------
heading("STEP 3 : Target classes")

# unacc = unacceptable, acc = acceptable, good, vgood = very good
print(df["class"].value_counts())

# ------------------------------------------------------------------
# STEP 4 : Encode text values into numbers
# ------------------------------------------------------------------
heading("STEP 4 : Encode categorical data")

encoders = {}                               # keep each encoder to decode later
for column in df.columns:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])   # e.g. low/med/high -> 1/2/0

print("Encoded Dataset:")
print(df.head())
print("\nClass labels:", dict(enumerate(encoders["class"].classes_)))

# ------------------------------------------------------------------
# STEP 5 : Separate features and target
# ------------------------------------------------------------------
heading("STEP 5 : Separate features and target")

X = df.drop("class", axis=1)               # 6 car features
y = df["class"]                            # target class
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

# ------------------------------------------------------------------
# STEP 6 : Split into training and testing sets
# ------------------------------------------------------------------
heading("STEP 6 : Split the dataset")

# stratify=y keeps the same class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ------------------------------------------------------------------
# STEP 7 : Create and train the Random Forest model
# ------------------------------------------------------------------
heading("STEP 7 : Train the Random Forest model")

rf = RandomForestClassifier(n_estimators=100, random_state=42)   # 100 decision trees
rf.fit(X_train, y_train)
print("Random Forest model trained successfully.")

# ------------------------------------------------------------------
# STEP 8 : Predict and evaluate
# ------------------------------------------------------------------
heading("STEP 8 : Prediction and evaluation")

y_pred = rf.predict(X_test)
print("Predicted values:", y_pred[:10])
print("Actual values   :", y_test.values[:10])

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f} %")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoders["class"].classes_))

cm = confusion_matrix(y_test, y_pred)      # rows = actual, columns = predicted
print("Confusion Matrix:")
print(cm)

# ------------------------------------------------------------------
# STEP 9 : Feature importance
# ------------------------------------------------------------------
heading("STEP 9 : Feature importance")

importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance.round(4))

# ------------------------------------------------------------------
# STEP 10 : Visualizations
# ------------------------------------------------------------------
heading("STEP 10 : Visualizations")

# 1. Confusion matrix
labels = encoders["class"].classes_
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="viridis")
for i in range(len(labels)):               # write the count inside each box
    for j in range(len(labels)):
        color = "black" if cm[i, j] > cm.max() / 2 else "white"   # readable on any colour
        plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()

# 2. Feature importance
plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()

print("Opening all graphs... close the graph windows to finish.")
plt.show()
