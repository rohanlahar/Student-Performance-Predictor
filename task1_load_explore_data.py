"""
Project 1: Student Performance Predictor
Task 1: Load and Explore the Data
-----------------------------------------------------------
Goal: Build a realistic dataset of "Hours Studied" vs "Exam Score"
for 60 students and explore it (structure, missing values, stats).

Note: This dataset is generated programmatically (not copied from any
textbook/sample). Study hours range from 0.5 to 12 hours, and scores
are generated using a realistic relationship (score grows with hours,
flattens near the top, and has natural random variation between
students) so the data resembles real exam results.

The cleaned dataset is saved to 'student_data.csv' so Task 2 and
Task 3 can reuse the exact same data.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Step 1: Generate an original dataset of 60 students
# ---------------------------------------------------------
np.random.seed(42)  # for reproducibility

n_students = 60
student_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_students + 1)]

# Hours studied: random values between 0.5 and 12 hours
hours_studied = np.round(np.random.uniform(0.5, 12, n_students), 1)

# Exam score: base relationship + diminishing returns + random noise
# (more hours generally means a higher score, but it tapers off near 100
#  and every student has some natural variation)
base_score = 30 + (hours_studied * 6.2) - (hours_studied ** 2) * 0.15
noise = np.random.normal(loc=0, scale=4, size=n_students)
exam_score = base_score + noise

# Clip scores so they stay within a realistic 0-100 range
exam_score = np.clip(exam_score, 0, 100)
exam_score = np.round(exam_score, 1)

df = pd.DataFrame({
    "StudentID": student_ids,
    "HoursStudied": hours_studied,
    "ExamScore": exam_score
})

# Intentionally introduce a couple of missing values, like real-world data
df.loc[5, "ExamScore"] = np.nan
df.loc[22, "HoursStudied"] = np.nan

# ---------------------------------------------------------
# Step 2: Explore the data
# ---------------------------------------------------------
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape (rows, columns):", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# ---------------------------------------------------------
# Step 3: Clean the data (handle missing values)
# ---------------------------------------------------------
# Fill missing HoursStudied / ExamScore with the column median
df["HoursStudied"] = df["HoursStudied"].fillna(df["HoursStudied"].median())
df["ExamScore"] = df["ExamScore"].fillna(df["ExamScore"].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Step 4: Save the cleaned dataset for Task 2
# ---------------------------------------------------------
output_path = "student_data.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset with {len(df)} students saved to '{output_path}'.")
print("This file will be used by Task 2 to train the model.")
