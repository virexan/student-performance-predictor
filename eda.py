import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("students.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# Performance Distribution
plt.figure(figsize=(8, 5))
df["performance"].value_counts().plot(kind="bar")
plt.title("Student Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("performance_distribution.png")
plt.show()

# Study Hours Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["study_hours"], bins=10)
plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("study_hours_distribution.png")
plt.show()

print("\nEDA Completed Successfully!")
print("Generated:")
print("- performance_distribution.png")
print("- study_hours_distribution.png")