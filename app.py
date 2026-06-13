import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("student_performance_model.pkl")

# Page Config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Performance Predictor")
st.write(
    "Predict student academic performance using Machine Learning."
)

st.divider()

# Inputs

study_hours = st.slider(
    "Study Hours Per Day",
    1,
    12,
    5
)

attendance = st.slider(
    "Attendance (%)",
    50,
    100,
    80
)

sleep_hours = st.slider(
    "Sleep Hours",
    4,
    10,
    7
)

assignments_completed = st.slider(
    "Assignments Completed",
    0,
    10,
    5
)

previous_score = st.slider(
    "Previous Score",
    0,
    100,
    70
)

# Prediction Button

if st.button("Predict Performance"):

    data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score]
    })

    prediction = model.predict(data)[0]

    performance_map = {
        0: "Average",
        1: "Excellent",
        2: "Good",
        3: "Poor"
    }

    result = performance_map[prediction]

    st.success(
        f"Predicted Performance: {result}"
    )

    if result == "Excellent":
        st.balloons()