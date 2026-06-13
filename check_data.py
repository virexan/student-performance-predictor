import pandas as pd

df = pd.read_csv("students.csv")

print(df["performance"].value_counts())