# predictor.py    

import onnxruntime as ort
import numpy as np
import pandas as pd
import os


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    return ort.InferenceSession(model_path)


def get_feature_matrix(df):
    feature_cols = [
        c for c in df.columns
        if c not in ["Date", "Time", "Datetime", "CO(GT)", "y_day", "y_week", "source"]
        and df[c].dtype != 'object'
    ]
    X = df[feature_cols].values.astype(np.float32)
    return X, feature_cols


def onnx_predict(session, X):
    input_name = session.get_inputs()[0].name
    y = session.run(None, {input_name: X})
    return y[0]


def predict_from_csv(
    preprocessed_csv,
    day_model="rf_day.onnx",
    week_model="rf_week.onnx"
):

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ONNX_DIR = os.path.join(PROJECT_ROOT, "outputs", "onnx_models")
    
    day_model_path = day_model if os.path.isabs(day_model) else os.path.join(ONNX_DIR, day_model)
    week_model_path = week_model if os.path.isabs(week_model) else os.path.join(ONNX_DIR, week_model)

    df = pd.read_csv(preprocessed_csv, sep=';')
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    X, feature_cols = get_feature_matrix(df)

    day_sess = load_model(day_model_path)
    week_sess = load_model(week_model_path)

    y_day_pred = onnx_predict(day_sess, X)
    y_week_pred = onnx_predict(week_sess, X)

    results = pd.DataFrame({
        "Datetime": df["Datetime"],
        "Actual_CO": df["CO(GT)"].values,
        "Pred_Day": y_day_pred.flatten(),
        "Pred_Week": y_week_pred.flatten()
    })
    
    return results