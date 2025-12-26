# data_preprocessor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def select_features(input_path, output_path):

    df = pd.read_csv(input_path, sep=';', decimal=',')
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop(columns=['NMHC(GT)', 'T', 'RH', 'AH', 'PT08.S1(CO)'], errors='ignore')

    if 'Date' in df.columns and 'Time' in df.columns:
        format_str = '%d/%m/%Y %H.%M.%S' if 'UCI' in input_path else '%Y-%m-%d %H:%M:%S'
        df['Datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format=format_str,
            errors='coerce'
        )
        df = df.drop(columns=['Date', 'Time'], errors='ignore')
    else:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    selected_features = [
        'Datetime',
        'hour_sin',
        'hour_cos',
        'dayofweek',
        'month',
        'C6H6(GT)',
        'PT08.S2(NMHC)',
        'PT08.S5(O3)',
        'NOx(GT)',
        'PT08.S3(NOx)',
        'NO2(GT)',
        'PT08.S4(NO2)'
    ]
    target = 'CO(GT)'
    final_df = df[selected_features + [target]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, sep=';', index=False)
    
    print("Selected dataset saved")
    print(f"File: {output_path}")
    print("\nFeatures:")
    for f in selected_features:
        print(" -", f)
    print("Target:", target)


def clean_missing_blocks(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, sep=';')
    
    ROW_MISSING_THRESHOLD = 50

    def clean_by_target(df, target_col, row_missing_threshold):
        print("\n" + "=" * 80)
        print(f"Cleaning based on: {target_col}")
        print("=" * 80)
        
        if target_col == "Datetime":
            df[target_col] = pd.to_datetime(df[target_col], errors="coerce")
            
        target_missing_mask = df[target_col].isna()
        df_target_missing = df[target_missing_mask].copy()
        
        print(f"Total rows where {target_col} is missing: {len(df_target_missing)}")
        
        num_bad_rows = 0
        bad_mask_full = pd.Series(False, index=df.index)
        
        if len(df_target_missing) == 0:
            print(f"No rows with missing {target_col}. Nothing to delete.")
        else:
            df_target_missing["row_missing_percent"] = df_target_missing.isna().mean(axis=1) * 100
            bad_rows = df_target_missing["row_missing_percent"] > row_missing_threshold
            num_bad_rows = bad_rows.sum()
            print(f"Rows with > {row_missing_threshold}% missing: {num_bad_rows}")
            
            bad_mask_full[target_missing_mask] = bad_rows.values
            
        return bad_mask_full, num_bad_rows

    mask_co, bad_co = clean_by_target(df.copy(), "CO(GT)", ROW_MISSING_THRESHOLD)
    mask_dt, bad_dt = clean_by_target(df.copy(), "Datetime", ROW_MISSING_THRESHOLD)
    
    combined_bad_mask = mask_co | mask_dt
    rows_to_delete = combined_bad_mask.sum()
    df_cleaned = df[~combined_bad_mask].copy()

    print("\n" + "=" * 80)
    print("Final cleaning summary (CO(GT) + Datetime)")
    print("=" * 80)
    print(f"Original total rows: {len(df)}")
    print(f"Rows deleted: {rows_to_delete}")
    print(f"Final rows: {len(df_cleaned)}")

    df_cleaned.to_csv(file_path, sep=';', index=False)
    print(f"Dataset cleaned and overwritten: {file_path}")

    summary_path = os.path.join(
        os.path.dirname(file_path),
        f"Cleaning_Summary_CO_and_Datetime.txt"
    )
    
    with open(summary_path, "w") as f:
        f.write(f"Cleaning Summary \n")
        f.write(f"Dataset: {os.path.basename(file_path)}\n")
        f.write(f"Original rows: {len(df)}\n")
        f.write(f"Rows deleted due to CO(GT): {bad_co}\n")
        f.write(f"Rows deleted due to Datetime: {bad_dt}\n")
        f.write(f"Total rows deleted: {rows_to_delete}\n")
        f.write(f"Final rows: {len(df_cleaned)}\n")
    
    print(f"Cleaning summary saved: {summary_path}")


def fill_missing_values(file_path):
    """پر کردن missing values با interpolation زمانی"""
    df = pd.read_csv(file_path, sep=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df = df.replace(-200, np.nan)

    cols_to_impute = [
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "PT08.S5(O3)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "CO(GT)",
    ]

    df[cols_to_impute] = (
        df[cols_to_impute]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    print(df[cols_to_impute].isna().sum())

    df_reset = df.reset_index()
    df_reset.to_csv(file_path, sep=';', index=False)
    print("Saved with time-based imputed features to:", file_path)


def final_advanced_preprocessing(input_path, output_final_path):

    df = pd.read_csv(input_path, sep=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    print("Original shape:", df.shape)
    print("Adding extended lags...")
    
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        df[f"CO_lag{lag}"] = df["CO(GT)"].shift(lag)

    print("Adding rolling features...")
    df['CO_roll_mean_3'] = df["CO(GT)"].rolling(3).mean()
    df['CO_roll_mean_7'] = df["CO(GT)"].rolling(7).mean()
    df['CO_roll_mean_24'] = df["CO(GT)"].rolling(24).mean()
    df['CO_roll_std_24'] = df["CO(GT)"].rolling(24).std()
    df['CO_roll_max_24'] = df["CO(GT)"].rolling(24).max()
    df['CO_roll_min_24'] = df["CO(GT)"].rolling(24).min()
    df['CO_trend_3h'] = df["CO(GT)"] - df["CO(GT)"].shift(3)
    df['CO_trend_24h'] = df["CO(GT)"] - df["CO(GT)"].shift(24)
    df['CO_NOx_ratio'] = df["CO(GT)"] / (df["NOx(GT)"] + 1)
    df['CO_NO2_ratio'] = df["CO(GT)"] / (df["NO2(GT)"] + 1)
    
    df["y_day"] = df["CO(GT)"].shift(-24)
    df["y_week"] = df["CO(GT)"].shift(-168)

    df = df.dropna().reset_index(drop=True)
    print("Shape after feature engineering:", df.shape)
    print(f"Added {len(df.columns) - 13} new features!")

    feature_cols = [col for col in df.columns
                    if col not in ["Datetime", "CO(GT)", "y_day", "y_week"]
                    and df[col].dtype in ['float64', 'int64']]
    
    print(f"Final features count: {len(feature_cols)}")

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("Advanced preprocessing completed!")
    
    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
    df.to_csv(output_final_path, sep=';', index=False)


def save_sample_for_testing(input_path, output_test_path, n_rows=100, method="first"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File '{input_path}' not found.")

    df = pd.read_csv(input_path, sep=';')

    if method == "first":
        df_sample = df.iloc[:n_rows].copy()
    elif method == "last":
        df_sample = df.iloc[-n_rows:].copy()
    else:
        raise ValueError("method must be last or first")

    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    df_sample.to_csv(output_test_path, sep=';', index=False)
    print(f"Sample of {n_rows} rows saved for testing: {output_test_path}")