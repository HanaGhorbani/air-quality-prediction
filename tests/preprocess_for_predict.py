# tests/preprocess_for_predict.py
import pandas as pd
import os

# Selcted Features for Training Models
top_features_day = [
    'hour_sin', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S5(O3)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'CO_lag1', 'CO_lag2', 'CO_lag3', 'CO_lag24', 'CO_lag48',
    'CO_lag72', 'CO_lag168', 'CO_roll_mean_3', 'CO_roll_mean_7',
    'CO_roll_mean_24', 'CO_trend_3h', 'CO_NO2_ratio'
]

top_features_week = [
    'hour_sin', 'month', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S5(O3)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'CO_lag1', 'CO_lag2', 'CO_lag3', 'CO_lag24', 'CO_lag48',
    'CO_lag72', 'CO_lag168', 'CO_roll_mean_3', 'CO_roll_mean_7',
    'CO_trend_3h', 'CO_NO2_ratio'
]

def preprocess_for_predict(
    input_csv,
    output_day_csv=None,
    output_week_csv=None
):

    if output_day_csv is None:
        output_day_csv = input_csv.replace(".csv", "_preprocessed_day.csv")
    if output_week_csv is None:
        output_week_csv = input_csv.replace(".csv", "_preprocessed_week.csv")

    df = pd.read_csv(input_csv, sep=';')


    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)


    df_day = df[['Datetime', 'CO(GT)'] + [f for f in top_features_day if f in df.columns]].copy()
    df_week = df[['Datetime', 'CO(GT)'] + [f for f in top_features_week if f in df.columns]].copy()


    os.makedirs(os.path.dirname(output_day_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_week_csv), exist_ok=True)

    df_day.to_csv(output_day_csv, sep=';', index=False)
    df_week.to_csv(output_week_csv, sep=';', index=False)

    print(f"Preprocessed DAY test data saved:   {output_day_csv} ({len(df_day)} rows)")
    print(f"Preprocessed WEEK test data saved:  {output_week_csv} ({len(df_week)} rows)")

    return output_day_csv, output_week_csv