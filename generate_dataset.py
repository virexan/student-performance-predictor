import pandas as pd
import numpy as np

np.random.seed(42)

num_students = 1000

study_hours = np.random.randint(1, 10, num_students)
attendance = np.random.randint(50, 100, num_students)
sleep_hours = np.random.randint(4, 10, num_students)
assignments_completed = np.random.randint(0, 11, num_students)
previous_score = np.random.randint(40, 100, num_students)

performance_score = (
    study_hours * 4
    + attendance * 0.4
    + sleep_hours * 2
    + assignments_completed * 3
    + previous_score * 0.3
)

performance = []
for score in performance_score:
    if score >= 95:
        performance.append("Excellent")
    elif score >= 80:
        performance.append("Good")
    elif score >= 65:
        performance.append("Average")
    else:
        performance.append("Poor")

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments_completed,
    "previous_score": previous_score,
    "performance": performance
})

df.to_csv("students.csv", index=False)

print("Dataset created successfully!")
print(df.head())