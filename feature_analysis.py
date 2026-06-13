import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("students.csv")

# Convert Performance to Numeric
performance_map = {
    "Poor": 0,
    "Average": 1,
    "Good": 2,
    "Excellent": 3
}

df["performance_numeric"] = df["performance"].map(performance_map)

# Correlation Matrix
correlation = df.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")
plt.show()

# Correlation with Target
target_corr = correlation["performance_numeric"].sort_values(ascending=False)

print("\nFeature Importance (Correlation Based)")
print("=" * 50)
print(target_corr)

print("\nHeatmap Saved Successfully!")