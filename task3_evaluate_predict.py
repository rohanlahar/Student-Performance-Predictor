"""
Project 1: Student Performance Predictor
Task 3: Evaluate and Make Predictions
-----------------------------------------------------------
Goal: Load the model trained in Task 2, evaluate it on the test
set using Mean Absolute Error (MAE) and R-squared, compare
predicted vs actual scores, and predict scores for new students.

Also generates a plot showing the full dataset, the regression line,
and the predictions for new students, saved as 'prediction_plot.png'.

Requires: 'exam_score_model.pkl', 'test_set.csv', and 'train_set.csv'
(all created by task2_train_model.py) must be in the same folder.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# Step 1: Load the trained model and the saved test set
# ---------------------------------------------------------
model = joblib.load("exam_score_model.pkl")
test_df = pd.read_csv("test_set.csv")
train_df = pd.read_csv("train_set.csv")

X_test = test_df[["HoursStudied"]]
y_test = test_df["ActualScore"]

print(f"Loaded trained model and {len(test_df)} test records.")

# ---------------------------------------------------------
# Step 2: Predict on the test data
# ---------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# Step 3: Evaluate performance
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.2f} points")
print(f"R-squared (R2 score): {r2:.2f}")
print("(Lower MAE is better. R2 closer to 1.0 means the model "
      "explains the data well.)")

# ---------------------------------------------------------
# Step 4: Compare actual vs predicted scores on the test set
# ---------------------------------------------------------
results = pd.DataFrame({
    "HoursStudied": test_df["HoursStudied"],
    "ActualScore": y_test.values,
    "PredictedScore": [round(p, 1) for p in y_pred]
})
results["Error"] = (results["ActualScore"] - results["PredictedScore"]).round(1)

print("\nActual vs Predicted scores on test set:")
print(results.to_string(index=False))

# ---------------------------------------------------------
# Step 5: Predict exam scores for brand-new students
# ---------------------------------------------------------
new_students = pd.DataFrame({"HoursStudied": [2.5, 4.5, 6.0, 9.5]})
new_predictions = model.predict(new_students)

print("\nPredictions for new students:")
for hours, score in zip(new_students["HoursStudied"], new_predictions):
    print(f"  Studied {hours} hours -> Predicted score: {score:.1f}")

# Save final results for reference
results.to_csv("evaluation_results.csv", index=False)
print("\nEvaluation results saved to 'evaluation_results.csv'.")

# ---------------------------------------------------------
# Step 6: Plot the data, the regression line, and predictions
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

# Training points
plt.scatter(train_df["HoursStudied"], train_df["ActualScore"],
            color="#4C72B0", alpha=0.7, label="Training data")

# Testing points
plt.scatter(test_df["HoursStudied"], test_df["ActualScore"],
            color="#DD8452", alpha=0.9, label="Testing data")

# Regression line across the full hours range
hours_range = np.linspace(0, 12, 100).reshape(-1, 1)
score_line = model.predict(pd.DataFrame(hours_range, columns=["HoursStudied"]))
plt.plot(hours_range, score_line, color="black", linewidth=2,
         label="Regression line")

# Highlight predictions for new students
plt.scatter(new_students["HoursStudied"], new_predictions,
            color="#55A868", marker="*", s=220, zorder=5,
            label="New student predictions")

plt.title("Student Performance Predictor: Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = "prediction_plot.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to '{plot_path}'.")
