import joblib
import pandas as pd

# Load Saved Model
model = joblib.load("student_performance_model.pkl")

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTOR")
print("=" * 50)

# User Input
study_hours = float(input("Study Hours Per Day: "))
attendance = float(input("Attendance (%): "))
sleep_hours = float(input("Sleep Hours: "))
assignments_completed = int(input("Assignments Completed (0-10): "))
previous_score = float(input("Previous Score: "))

# Create DataFrame
student_data = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "sleep_hours": [sleep_hours],
    "assignments_completed": [assignments_completed],
    "previous_score": [previous_score]
})

# Predict
prediction = model.predict(student_data)[0]

# Decode Prediction
performance_map = {
    0: "Average",
    1: "Excellent",
    2: "Good",
    3: "Poor"
}

result = performance_map[prediction]

print("\n" + "=" * 50)
print(f"Predicted Performance: {result}")
print("=" * 50)