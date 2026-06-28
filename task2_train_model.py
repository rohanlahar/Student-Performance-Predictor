"""
Project 1: Student Performance Predictor
Task 2: Train a Linear Regression Model
-----------------------------------------------------------
Goal: Load the dataset created in Task 1, split it into training
and testing sets, and train a linear regression model that learns
the relationship between hours studied and exam score.

Requires: 'student_data.csv' (created by task1_load_explore_data.py)
must be in the same folder.

This script saves the trained model to 'exam_score_model.pkl' so
Task 3 can load it and run evaluation/predictions without retraining.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Step 1: Load the dataset produced by Task 1
# ---------------------------------------------------------
df = pd.read_csv("student_data.csv")
print(f"Loaded {len(df)} student records from 'student_data.csv'.")
print(df.head())

# ---------------------------------------------------------
# Step 2: Define features (X) and target (y)
# ---------------------------------------------------------
X = df[["HoursStudied"]]   # double brackets -> keeps it as a DataFrame
y = df["ExamScore"]

# ---------------------------------------------------------
# Step 3: Split into training (80%) and testing (20%) sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {len(X_train)} students")
print(f"Testing set size: {len(X_test)} students")

# ---------------------------------------------------------
# Step 4: Create and train the Linear Regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_

print("\nModel trained successfully.")
print(f"Slope (coefficient): {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"\nInterpretation: each extra hour studied changes the predicted "
      f"score by about {slope:.2f} points. With 0 hours studied, the "
      f"baseline predicted score is about {intercept:.2f}.")

# ---------------------------------------------------------
# Step 5: Save the trained model and the train/test split
#         so Task 3 can evaluate it without retraining
# ---------------------------------------------------------
joblib.dump(model, "exam_score_model.pkl")

X_test.assign(ActualScore=y_test).to_csv("test_set.csv", index=False)
X_train.assign(ActualScore=y_train).to_csv("train_set.csv", index=False)

print("\nModel saved to 'exam_score_model.pkl'.")
print("Test set saved to 'test_set.csv' for use in Task 3.")
