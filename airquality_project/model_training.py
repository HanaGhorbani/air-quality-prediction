# model_training.py
import pandas as pd
import numpy as np
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from visualization import (
    plot_colored_results_table,
    plot_metric_bar,
    plot_scatter_predictions,
    plot_residuals,
    plot_predictions_over_time,
    plot_model_correlation_heatmap
)

SEED = 42
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def manage_outliers(df, cols_to_check=None, method='clip', iqr_multiplier=1.5):
    print("\nHandling outliers...")
    if cols_to_check is None:
        cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols_to_check:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        if method == 'clip':
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif method == 'remove':
            df = df[(df[col] >= lower) & (df[col] <= upper)].copy()

    print("Outlier handling completed.")
    return df


def evaluate_model(y_true, y_pred, name=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(
        f"{name:<25} "
        f"MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | "
        f"R²={r2:.4f} | MAPE={mape:.2f}%"
    )
    return {
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    }


def save_to_onnx(model, n_features, filename):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ONNX_DIR = os.path.join(PROJECT_ROOT, "outputs", "onnx_models")
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    full_path = os.path.join(ONNX_DIR, filename)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=13)
    
    with open(full_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Model exported to ONNX: {full_path}")


def train_and_evaluate(
    data_path="AirQuality_Combined_Clean.csv",
    outlier_method='clip',
    iqr_multiplier=1.5
):
    print("=" * 100)
    print("FINAL MODELING (Linear Regression + Random Forest)")
    print("=" * 100)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_results")
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(data_path, sep=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df = manage_outliers(df, method=outlier_method, iqr_multiplier=iqr_multiplier)

    n = len(df)
    train_end, val_end = int(0.7 * n), int(0.85 * n)
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    feature_cols = [
        col for col in df.columns
        if col not in ["Datetime", "CO(GT)", "y_day", "y_week", "source"]
        and df[col].dtype != 'object'
    ]

    X_train, X_val, X_test = (
        train[feature_cols].values,
        val[feature_cols].values,
        test[feature_cols].values
    )

    y_day_train, y_week_train = train["y_day"], train["y_week"]
    y_day_test, y_week_test = test["y_day"], test["y_week"]
    datetime_test = test["Datetime"]

    results = []

    print("\n1. Persistence Baseline")
    persist_day_metrics = evaluate_model(y_day_test, test["CO_lag24"].values, "Day-ahead (lag24)")
    persist_week_metrics = evaluate_model(y_week_test, test["CO_lag168"].values, "Week-ahead (lag168)")
    results.append({
        'Model': 'Persistence',
        **persist_day_metrics,
        **{f'Week_{k}': v for k, v in persist_week_metrics.items()}
    })


    print("\n2. Feature Selection (Top 20)")
    selector_day = SelectKBest(score_func=f_regression, k=20)
    selector_week = SelectKBest(score_func=f_regression, k=20)
    
    selector_day.fit(X_train, y_day_train)
    top_features_day = [feature_cols[i] for i in selector_day.get_support(indices=True)]
    print("Top 20 features (day):", top_features_day)
    
    selector_week.fit(X_train, y_week_train)
    top_features_week = [feature_cols[i] for i in selector_week.get_support(indices=True)]
    print("Top 20 features (week):", top_features_week)

    X_train_day_sel = selector_day.fit_transform(X_train, y_day_train)
    X_test_day_sel = selector_day.transform(X_test)

    X_train_week_sel = selector_week.fit_transform(X_train, y_week_train)
    X_test_week_sel = selector_week.transform(X_test)

    print("\n3. Linear Regression")
    lr_day = LinearRegression().fit(X_train_day_sel, y_day_train)
    lr_week = LinearRegression().fit(X_train_week_sel, y_week_train)

    y_day_pred_lr = lr_day.predict(X_test_day_sel)
    y_week_pred_lr = lr_week.predict(X_test_week_sel)

    lr_day_metrics = evaluate_model(y_day_test, y_day_pred_lr, "Day-ahead (Linear)")
    lr_week_metrics = evaluate_model(y_week_test, y_week_pred_lr, "Week-ahead (Linear)")

    results.append({
        'Model': 'Linear Regression',
        **lr_day_metrics,
        **{f'Week_{k}': v for k, v in lr_week_metrics.items()}
    })

    print("\n4. Random Forest")
    rf_day = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_week = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

    rf_day.fit(X_train_day_sel, y_day_train)
    rf_week.fit(X_train_week_sel, y_week_train)

    y_day_pred_rf = rf_day.predict(X_test_day_sel)
    y_week_pred_rf = rf_week.predict(X_test_week_sel)

    rf_day_metrics = evaluate_model(y_day_test, y_day_pred_rf, "Day-ahead (RF)")
    rf_week_metrics = evaluate_model(y_week_test, y_week_pred_rf, "Week-ahead (RF)")

    results.append({
        'Model': 'Random Forest',
        **rf_day_metrics,
        **{f'Week_{k}': v for k, v in rf_week_metrics.items()}
    })


    results_df = pd.DataFrame(results).round(4)
    results_path = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nAll evaluation metrics saved to: {results_path}")

    save_to_onnx(lr_day, len(top_features_day), "lr_day.onnx")
    save_to_onnx(lr_week, len(top_features_week), "lr_week.onnx")
    save_to_onnx(rf_day, len(top_features_day), "rf_day.onnx")
    save_to_onnx(rf_week, len(top_features_week), "rf_week.onnx")

    plot_colored_results_table(results_df)
    plot_metric_bar(results_df, metric_day="RMSE", metric_week="Week_RMSE",
                    title="RMSE Comparison", filename="02_rmse_comparison.png")
    plot_metric_bar(results_df, metric_day="MAE", metric_week="Week_MAE",
                    title="MAE Comparison", filename="03_mae_comparison.png")
    plot_metric_bar(results_df, metric_day="R2", metric_week="Week_R2",
                    title="R² Comparison", filename="04_r2_comparison.png")

    y_pred_day_dict = {
        "Linear Regression": y_day_pred_lr,
        "Random Forest": y_day_pred_rf
    }
    y_pred_week_dict = {
        "Linear Regression": y_week_pred_lr,
        "Random Forest": y_week_pred_rf
    }

    plot_scatter_predictions(y_day_test, y_pred_day_dict, horizon="Day", filename="05_scatter_day.png")
    plot_scatter_predictions(y_week_test, y_pred_week_dict, horizon="Week", filename="06_scatter_week.png")

    plot_residuals(y_day_test, y_pred_day_dict, horizon="Day", filename="07_residuals_day.png")
    plot_residuals(y_week_test, y_pred_week_dict, horizon="Week", filename="08_residuals_week.png")

    plot_predictions_over_time(datetime_test, {"Actual": y_day_test, **y_pred_day_dict}, 
                              horizon="Day", filename="09_timeseries_day.png")
    plot_predictions_over_time(datetime_test, {"Actual": y_week_test, **y_pred_week_dict}, 
                              horizon="Week", filename="10_timeseries_week.png")

    plot_model_correlation_heatmap(y_pred_day_dict, horizon="Day", filename="11_corr_day.png")
    plot_model_correlation_heatmap(y_pred_week_dict, horizon="Week", filename="12_corr_week.png")

    print("\nAll visualizations saved in outputs/figures/")
    return results_df