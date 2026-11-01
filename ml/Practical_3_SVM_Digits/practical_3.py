"""
=====================================================================
 ML PRACTICAL 3
 Title   : Classification Analysis using SVM
           (Support Vector Machine)
 Aim     : Classify images of handwritten digits (0 to 9) with SVM.
 Dataset : Digits dataset - built into scikit-learn, so no file is
           needed (1797 images, each 8 x 8 pixels)
=====================================================================
"""

import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")          # hide library warnings for clean output


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("ML PRACTICAL 3 : CLASSIFICATION OF HANDWRITTEN DIGITS USING SVM")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the digits dataset")

digits = load_digits(n_class=10)
print("Data shape   :", digits.data.shape, "-> 1797 images, 64 pixels each")
print("Images shape :", digits.images.shape, "-> each image is 8 x 8")
print("Classes      :", list(digits.target_names))
print("First 9 labels:", list(digits.target[0:9]))

print("\nPixel values of the first image (label = 0):")
print(digits.images[0])        # 0 = white, 16 = black

# ------------------------------------------------------------------
# STEP 2 : Split data into training and testing parts
# ------------------------------------------------------------------
heading("STEP 2 : Split into training and testing data")

main_data = digits.data
targets = digits.target

X_train, y_train = main_data[:1500], targets[:1500]    # first 1500 images to train
X_test, y_test = main_data[1500:], targets[1500:]      # remaining 297 images to test
print("Training images:", len(X_train))
print("Testing images :", len(X_test))

# ------------------------------------------------------------------
# STEP 3 : Train the SVM classifier
# ------------------------------------------------------------------
heading("STEP 3 : Train the SVM classifier")

# gamma : how closely the model follows each training point
# C     : penalty for wrong classification (higher = stricter)
svc = svm.SVC(gamma=0.001, C=100)
svc.fit(X_train, y_train)
print("SVM model trained successfully (gamma=0.001, C=100).")

# ------------------------------------------------------------------
# STEP 4 : Predict and check accuracy
# ------------------------------------------------------------------
heading("STEP 4 : Predictions and accuracy")

predictions = svc.predict(X_test)
comparison = pd.DataFrame({"Predicted": predictions, "Actual": y_test})
print("First 20 predictions:")
print(comparison.head(20).T.to_string())

wrong = (predictions != y_test).sum()
print(f"\nCorrect predictions : {len(y_test) - wrong} / {len(y_test)}")
print(f"Accuracy            : {accuracy_score(y_test, predictions) * 100:.2f} %")

# ------------------------------------------------------------------
# STEP 5 : Confusion matrix and classification report
# ------------------------------------------------------------------
heading("STEP 5 : Confusion matrix and classification report")

cm = confusion_matrix(y_test, predictions)     # rows = actual, columns = predicted
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ------------------------------------------------------------------
# STEP 6 : Visualizations
# ------------------------------------------------------------------
heading("STEP 6 : Visualizations")

# 1. First 64 digit images
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap="binary")
    axi.set(xticks=[], yticks=[])
fig.suptitle("First 64 Handwritten Digits")

# 2. All digits squeezed from 64 dimensions to 2 with Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
plt.figure(figsize=(8, 6))
plt.scatter(projection[:, 0], projection[:, 1], c=digits.target, cmap="viridis", s=10)
plt.colorbar(ticks=range(10), label="Digit value")
plt.title("Digits Clustered with Isomap (64D -> 2D)")

# 3. One digit up close
plt.figure(figsize=(4, 4))
plt.imshow(digits.images[4], cmap=plt.cm.gray_r)
plt.title("Original digit is: " + str(digits.target[4]))

# 4. Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

# 5. Some test images with predicted labels
fig, ax = plt.subplots(2, 8, figsize=(12, 4))
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(8, 8), cmap="binary")
    color = "green" if predictions[i] == y_test[i] else "red"   # red = wrong
    axi.set_title(f"Pred: {predictions[i]}", color=color)
    axi.set(xticks=[], yticks=[])
fig.suptitle("Test Images with SVM Predictions")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
