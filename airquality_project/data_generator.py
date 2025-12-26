# data_generator.py
import pandas as pd
import numpy as np
import os


def generate_synthetic_data(output_path):

    print("Generating Synthetic AIR QUALITY Datawith realistic missing & outliers...")
    
    time_synth = pd.date_range("2025-01-01", periods=5000, freq="H")
    date_col = time_synth.date
    time_col = time_synth.time
    hour = time_synth.hour
    dayofweek = time_synth.dayofweek
    np.random.seed(42)

    CO_GT = 1.6 + 0.9 * np.sin(2*np.pi*hour/24) + np.random.normal(0, 0.3, 5000)
    NMHC_GT = 150 + 60 * np.sin(2*np.pi*dayofweek/7) + np.random.normal(0, 20, 5000)
    C6H6_GT = 2.4 + 0.7 * np.cos(2*np.pi*hour/24) + np.random.normal(0, 0.4, 5000)
    NOx_GT = 50 + 30 * np.sin(2*np.pi*hour/24) + np.random.normal(0, 10, 5000)
    NO2_GT = 30 + 18 * np.cos(2*np.pi*hour/24) + np.random.normal(0, 6, 5000)

    PT08_S1_CO = 1100 + 500 * CO_GT + np.random.normal(0, 80, 5000)
    PT08_S2_NMHC = 1200 + 420 * C6H6_GT + np.random.normal(0, 90, 5000)
    PT08_S3_NOx = 950 + 260 * NOx_GT + np.random.normal(0, 70, 5000)
    PT08_S4_NO2 = 1150 + 300 * NO2_GT + np.random.normal(0, 75, 5000)
    PT08_S5_O3 = 850 + 220 * (1 - np.cos(2*np.pi*hour/24)) + np.random.normal(0, 60, 5000)

    T = 15 + 10 * np.sin(2*np.pi*hour/24) + np.random.normal(0, 1.5, 5000)
    RH = 55 + 20 * np.cos(2*np.pi*hour/24) + np.random.normal(0, 5, 5000)
    AH = 0.6 + 0.3 * np.sin(2*np.pi*hour/24) + np.random.normal(0, 0.05, 5000)

    df = pd.DataFrame({
        'Date': date_col,
        'Time': time_col,
        'CO(GT)': np.clip(CO_GT, 0, 5),
        'PT08.S1(CO)': np.clip(PT08_S1_CO, 500, 2000),
        'NMHC(GT)': np.clip(NMHC_GT, 0, 500),
        'C6H6(GT)': np.clip(C6H6_GT, 0, 10),
        'PT08.S2(NMHC)': np.clip(PT08_S2_NMHC, 200, 2000),
        'NOx(GT)': np.clip(NOx_GT, 0, 200),
        'PT08.S3(NOx)': np.clip(PT08_S3_NOx, 400, 2000),
        'NO2(GT)': np.clip(NO2_GT, 0, 100),
        'PT08.S4(NO2)': np.clip(PT08_S4_NO2, 500, 2000),
        'PT08.S5(O3)': np.clip(PT08_S5_O3, 300, 1700),
        'T': T,
        'RH': RH,
        'AH': AH
    })

    # ========================================
    # Missing Values
    # ========================================
    numeric_cols = [col for col in df.columns if col not in ['Date', 'Time']]

    missing_rates = {
        'CO(GT)': 0.18,      
        'NOx(GT)': 0.18,
        'NO2(GT)': 0.18,
        'C6H6(GT)': 0.09,
        'PT08.S1(CO)': 0.02,
        'PT08.S2(NMHC)': 0.02,
        'PT08.S3(NOx)': 0.02,
        'PT08.S4(NO2)': 0.02,
        'PT08.S5(O3)': 0.02,
        'T': 0.005,
        'RH': 0.005,
        'AH': 0.005,
        'NMHC(GT)': 0.90      
    }

    for col, rate in missing_rates.items():
        if col in df.columns:
            mask = np.random.rand(len(df)) < rate
            df.loc[mask, col] = np.nan

    # ========================================
    # Outliers
    # ========================================
    for col in numeric_cols:
        if col in ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']:

            outlier_rate = 0.04  
            mask_high = np.random.rand(len(df)) < outlier_rate / 2
            mask_low = np.random.rand(len(df)) < outlier_rate / 2
            
            mean = df[col].mean()
            std = df[col].std()
            df.loc[mask_high, col] = df[col].clip(lower=0).max() * np.random.uniform(1.3, 1.8, mask_high.sum())
            df.loc[mask_low, col] = np.random.uniform(mean - 3*std, mean - 1.5*std, mask_low.sum())
        
        elif 'PT08' in col:
            outlier_mask = np.random.rand(len(df)) < 0.035
            df.loc[outlier_mask, col] = df[col].max() * np.random.uniform(1.4, 2.2, outlier_mask.sum())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"SAVED: {output_path} ({len(df)} rows)")
    print(f"Missing rates per column:")
    print(df[numeric_cols].isna().mean().round(3))
    print("\nOutliers added to pollutants and sensor readings.")
    print("Features match original AirQuality dataset exactly")



