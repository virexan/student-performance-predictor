import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load Data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("=" * 50)
print("RANDOM FOREST TRAINING")
print("=" * 50)

# Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Predict
predictions = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# --------------------------
# Confusion Matrix
# --------------------------

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

# --------------------------
# Feature Importance
# --------------------------

importance = rf.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_df)

plt.figure(figsize=(8,5))

plt.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

print("\nGenerated Files:")
print("- confusion_matrix.png")
print("- feature_importance.png")