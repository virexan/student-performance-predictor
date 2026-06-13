import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("students.csv")

print("=" * 50)
print("PREPROCESSING DATA")
print("=" * 50)

# Encode Target Variable
label_encoder = LabelEncoder()

df["performance"] = label_encoder.fit_transform(df["performance"])

print("\nEncoded Classes:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {index}")

# Features and Target
X = df.drop("performance", axis=1)
y = df["performance"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Save Processed Files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nFiles Saved Successfully!")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")