import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

# Load Data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

# Train Best Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "student_performance_model.pkl")

print("=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print("\nFile Created:")
print("student_performance_model.pkl")