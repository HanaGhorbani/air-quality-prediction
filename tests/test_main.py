# test_main.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from preprocess_for_predict import preprocess_for_predict
from predictor import predict_from_csv

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# Paths 
# =====================================================
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "cleaned_data")
OUTPUT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
ONNX_MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "onnx_models")

TEST_DATA_CSV = os.path.join(TEST_DATA_DIR, "test_data.csv")

TEST_PREPROCESSED_DAY = os.path.join(TEST_DATA_DIR, "test_preprocessed_day.csv")
TEST_PREPROCESSED_WEEK = os.path.join(TEST_DATA_DIR, "test_preprocessed_week.csv")

OUT_PRED_DAY = os.path.join(TEST_DATA_DIR, "test_predictions_day.csv")
OUT_PRED_WEEK = os.path.join(TEST_DATA_DIR, "test_predictions_week.csv")

DAY_MODEL_PATH = os.path.join(ONNX_MODELS_DIR, "rf_day.onnx")
WEEK_MODEL_PATH = os.path.join(ONNX_MODELS_DIR, "rf_week.onnx")

# =====================================================
# STEP 1 — Preprocess for day & week models
# =====================================================
print("\nSTEP 1 — Preprocessing test data...")
preprocess_for_predict(
    input_csv=TEST_DATA_CSV,
    output_day_csv=TEST_PREPROCESSED_DAY,
    output_week_csv=TEST_PREPROCESSED_WEEK
)

# =====================================================
# STEP 2 — Predict using ONNX models
# =====================================================
print("\nSTEP 2 — Running ONNX predictions...")
preds_day = predict_from_csv(
    preprocessed_csv=TEST_PREPROCESSED_DAY,
    day_model=DAY_MODEL_PATH,
    week_model=WEEK_MODEL_PATH
)

preds_week = predict_from_csv(
    preprocessed_csv=TEST_PREPROCESSED_WEEK,
    day_model=DAY_MODEL_PATH,
    week_model=WEEK_MODEL_PATH
)

# =====================================================
# STEP 3 — Save predictions
# =====================================================
preds_day.to_csv(OUT_PRED_DAY, index=False)
preds_week.to_csv(OUT_PRED_WEEK, index=False)
print(f"\nPredictions saved:\n  • {OUT_PRED_DAY}\n  • {OUT_PRED_WEEK}")

# =====================================================
# STEP 4 — Evaluate performance
# =====================================================
print("\nSTEP 4 — Evaluating model performance...")
def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{label:20} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

print("\n" + "="*60)
print("DAY Model Performance")
print("="*60)
evaluate(preds_day["Actual_CO"].values, preds_day["Pred_Day"].values, "Day-ahead")

print("\n" + "="*60)
print("WEEK Model Performance")
print("="*60)
evaluate(preds_week["Actual_CO"].values, preds_week["Pred_Week"].values, "Week-ahead")

# =====================================================
# STEP 5 — Save predictions plots
# =====================================================
def save_predictions_plot(actual, pred, output_filename, title="Predictions vs Actual"):
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual CO", marker='o', markersize=4)
    plt.plot(pred, label="Predicted CO", marker='x', markersize=4)
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("CO(GT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_FIGURES_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")

print("\nSaving prediction plots...")
save_predictions_plot(
    preds_day["Actual_CO"].values, 
    preds_day["Pred_Day"].values,
    "test_day_ahead_predictions.png", 
    "Test Set - Day-ahead CO Predictions"
)

save_predictions_plot(
    preds_week["Actual_CO"].values, 
    preds_week["Pred_Week"].values,
    "test_week_ahead_predictions.png", 
    "Test Set - Week-ahead CO Predictions"
)

print("\nTest execution completed!")