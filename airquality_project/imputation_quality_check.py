# imputation_quality_check.py
import pandas as pd
import numpy as np
import math
import os


cols_to_check = [
    "CO(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "PT08.S5(O3)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
]

def evaluate_imputation_quality(raw_path, clean_path, dataset_name="Real"):

    print("\n" + "="*80)
    print(f"Imputation Quality Evaluation (Metrics Only) - {dataset_name}")
    print("="*80)

    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found: {raw_path}")
        return
    
    if not os.path.exists(clean_path):
        print(f"Error: Clean file not found: {clean_path}")
        return

    raw = pd.read_csv(raw_path, sep=';', decimal=',')
    clean = pd.read_csv(clean_path, sep=';')

    raw = raw.loc[:, ~raw.columns.str.contains("^Unnamed")]
    clean = clean.loc[:, ~clean.columns.str.contains("^Unnamed")]

    if 'Date' in raw.columns and 'Time' in raw.columns:
        raw["Datetime"] = pd.to_datetime(
            raw["Date"] + " " + raw["Time"],
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce"
        )
        raw = raw.set_index("Datetime").sort_index()

    n = min(len(raw), len(clean))
    raw = raw.iloc[:n]
    clean = clean.iloc[:n]

    raw = raw.replace(-200, np.nan)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "analyses", "imputation_quality")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for col in cols_to_check:
        if col not in raw.columns or col not in clean.columns:
            print(f"Column {col} not found → skipped")
            continue

        print(f"\nProcessing feature: {col}")

        raw_series = pd.to_numeric(raw[col], errors="coerce")

        valid_idx = raw_series.dropna().index

        if len(valid_idx) > 10:
            sample_size = max(5, int(0.05 * len(valid_idx)))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(valid_idx, size=sample_size, replace=False)

            test_series = raw_series.copy()
            test_series.loc[sample_idx] = np.nan

            test_imputed = (
                test_series
                .interpolate(method="time")
                .ffill()
                .bfill()
            )

            y_true = raw_series.loc[sample_idx]
            y_pred = test_imputed.loc[sample_idx]

            mae = (y_true - y_pred).abs().mean()
            rmse = math.sqrt(((y_true - y_pred) ** 2).mean())

            print(f"  Back-test results:")
            print(f"    MAE:  {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Tested samples: {sample_size}")

            results.append({
                "feature": col,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "tested_samples": sample_size
            })
        else:
            print(f"  Not enough valid raw values for back-test")


    if results:
        print("\n" + "="*80)
        print(f"Final Back-test Summary - {dataset_name}")
        print("="*80)
        
        res_df = pd.DataFrame(results)
        print(res_df)

        report_filename = f"{dataset_name}_Imputation_Backtest_Summary.csv"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        
        res_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\nReport saved successfully: {report_path}")
    else:
        print("No valid results for any feature.")