# Visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import dataframe_image as dfi


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_colored_results_table(results_df, filename="01_results_table.png"):
    df_plot = results_df.copy()
    df_plot.set_index('Model', inplace=True)

    def highlight_best(s):
        if "R2" in s.name:
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return ["background-color: yellow" if v else "" for v in is_best]

    styled = df_plot.style.apply(highlight_best)

    filepath = os.path.join(FIGURES_DIR, filename)
    try:
        dfi.export(styled, filepath)
        print(f"Results table saved: {filepath}")
    except Exception as e:
        print("Error saving results table with dataframe_image:", e)
        print("Skipping table image export.")


def plot_metric_bar(results_df, metric_day="RMSE", metric_week="Week_RMSE",
                   title="Metric Comparison", filename="metric_comparison.png"):
    df_plot = results_df.copy()
    df_plot = df_plot[['Model', metric_day, metric_week]]

    plt.figure(figsize=(10,6))
    x = np.arange(len(df_plot['Model']))
    width = 0.35

    plt.bar(x - width/2, df_plot[metric_day], width, label="Day-ahead")
    plt.bar(x + width/2, df_plot[metric_week], width, label="Week-ahead")

    plt.xticks(x, df_plot['Model'], rotation=45)
    plt.ylabel(metric_day)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved: {out_path}")


def plot_scatter_predictions(y_true, y_pred_dict, horizon="Day", filename="scatter_predictions.png"):
    plt.figure(figsize=(8,8))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label="Perfect")
    for model_name, y_pred in y_pred_dict.items():
        plt.scatter(y_true, y_pred, alpha=0.5, label=model_name)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{horizon}-ahead: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved: {out_path}")


def plot_residuals(y_true, y_pred_dict, horizon="Day", filename="residuals.png"):
    plt.figure(figsize=(10,6))
    for model_name, y_pred in y_pred_dict.items():
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, label=model_name)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{horizon}-ahead Residuals")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual plot saved: {out_path}")


def plot_predictions_over_time(df_time, y_true_dict, horizon="Day", filename="predictions_over_time.png"):
    plt.figure(figsize=(12,6))
    for label, series in y_true_dict.items():
        plt.plot(df_time, series, label=label)
    plt.xlabel("Datetime")
    plt.ylabel("CO(GT)")
    plt.title(f"{horizon}-ahead Predictions over Time")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved: {out_path}")


def plot_model_correlation_heatmap(y_pred_dict, horizon="Day", filename="model_correlation.png"):
    df_pred = pd.DataFrame(y_pred_dict)
    corr = df_pred.corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{horizon}-ahead Prediction Correlation")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved: {out_path}")