# data_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def analyze_dataset(input_path, output_dir, timestamp_prefix):
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting initial analysis for: {input_path}")
    print(f"Prefix: {timestamp_prefix}")

    df = pd.read_csv(input_path, sep=';', decimal=',')
    df = df.replace(-200, np.nan)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print(f"WARNING: No numeric columns found in {input_path}")
        print("Available columns:", df.columns.tolist())
        return

    print(f"Numeric columns detected ({len(numeric_cols)}): {numeric_cols}")


    missing_percent = []
    outlier_percent = []

    for col in numeric_cols:
        missing_pct = df[col].isna().mean() * 100
        missing_percent.append(missing_pct)

        series = df[col].dropna()
        if len(series) > 5:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = series[(series < lower) | (series > upper)]
            outlier_pct = (len(outliers) / len(df)) * 100
            outlier_percent.append(outlier_pct)
        else:
            outlier_percent.append(0.0)
            print(f"Warning: {col} has too few valid values for outlier detection")


    summary_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Missing (%)': missing_percent,
        'Outlier (%)': outlier_percent
    })
    summary_df['Total Problem (%)'] = summary_df['Missing (%)'] + summary_df['Outlier (%)']
    summary_df = summary_df.sort_values('Total Problem (%)').reset_index(drop=True)


    plt.figure(figsize=(16, 8))
    x_pos = np.arange(len(summary_df))
    width = 0.38
    plt.bar(x_pos - width/2, summary_df['Outlier (%)'], width, label='Outliers (%)', color='#e74c3c', alpha=0.85)
    plt.bar(x_pos + width/2, summary_df['Missing (%)'], width, label='Missing (%)', color='#3498db', alpha=0.85)
    plt.xticks(x_pos, summary_df['Feature'], rotation=60, ha='right', fontsize=9)
    plt.ylabel('Percentage (%)')
    plt.title(f'Missing Values vs Outliers per Feature - {timestamp_prefix}', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    
    bar_plot_path = os.path.join(output_dir, f"{timestamp_prefix}_Missing_vs_Outlier_BarPlot.png")
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved: {bar_plot_path}")


    valid_cols = [col for col in numeric_cols 
                  if df[col].notna().sum() >= 10 and df[col].std() > 1e-6]
    
    if len(valid_cols) >= 2:
        corr_matrix = df[valid_cols].corr()
        if not corr_matrix.empty and not corr_matrix.isna().all().all():
            plt.figure(figsize=(15, 12))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        linewidths=0.6, linecolor='gray', cbar_kws={"shrink": 0.7})
            plt.xticks(rotation=60, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.title(f'Correlation Matrix - {timestamp_prefix}', fontsize=16, pad=20)
            plt.tight_layout()
            
            heatmap_path = os.path.join(output_dir, f"{timestamp_prefix}_Correlation_Heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Correlation heatmap saved: {heatmap_path}")
        else:
            print(f"Skipping correlation heatmap: matrix invalid or all NaN")
    else:
        print(f"Skipping correlation heatmap: not enough valid columns")


    print("\nGenerating Emptiness (Missing Co-occurrence) Matrix...")
    n = len(numeric_cols)
    emptiness_matrix = pd.DataFrame(np.zeros((n, n)), index=numeric_cols, columns=numeric_cols)

    for i, col_i in enumerate(numeric_cols):
        missing_rows = df[col_i].isna()
        if missing_rows.sum() < 1:
            continue
        for j, col_j in enumerate(numeric_cols):
            emptiness_matrix.iloc[i, j] = df[col_j][missing_rows].isna().mean() * 100

    emptiness_matrix = emptiness_matrix.round(1)

    plt.figure(figsize=(14, 11))
    sns.heatmap(emptiness_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray',
                cbar_kws={"shrink": 0.8, "label": "Missing % when row feature missing"})
    plt.title(f'Emptiness Matrix - {timestamp_prefix}', fontsize=16, pad=20)
    plt.xlabel('Feature Y (conditional missing %)', fontsize=12)
    plt.ylabel('Feature X (given missing)', fontsize=12)
    plt.xticks(rotation=60, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    emptiness_plot_path = os.path.join(output_dir, f"{timestamp_prefix}_Emptiness_Cooccurrence_Matrix.png")
    plt.savefig(emptiness_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Emptiness Matrix saved: {emptiness_plot_path}")

    if 'CO(GT)' in numeric_cols:
        print("\n" + "="*70)
        print("Row-level Missing Analysis for rows where CO(GT) is missing")
        print("="*70)
        TARGET_COL = 'CO(GT)'
        ROW_MISSING_THRESHOLD = 50

        co_missing_df = df[df[TARGET_COL].isna()].copy()
        print(f"Total rows where {TARGET_COL} is missing: {len(co_missing_df)}")

        if len(co_missing_df) > 0:
            co_missing_df['row_missing_percent'] = co_missing_df.isna().mean(axis=1) * 100
            co_missing_df['high_missing_row'] = (co_missing_df['row_missing_percent'] > ROW_MISSING_THRESHOLD).astype(int)

            total = len(co_missing_df)
            high = co_missing_df['high_missing_row'].sum()
            print(f"Rows with > {ROW_MISSING_THRESHOLD}% missing: {high} ({high/total*100:.2f}%)")

            row_report_path = os.path.join(output_dir, f"{timestamp_prefix}_CO_Row_Missing_Analysis.csv")
            co_missing_df[['row_missing_percent', 'high_missing_row']].to_csv(row_report_path, index=False)
            print(f"Row-level report saved: {row_report_path}")

    print("\n" + "="*70)
    print(f"Initial analysis completed for {timestamp_prefix}")
    print(f"All outputs saved in: {output_dir}")
    print("="*70)


def analyze_clean_dataset(input_path, output_dir, timestamp_prefix):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, sep=';')
    df = df.replace([-200, 'NaN'], np.nan)


    for col in df.columns:
        if col != 'Datetime' and col != 'source':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("No numeric columns in clean dataset")
        return

    missing_percent = [df[col].isna().mean() * 100 for col in numeric_cols]
    outlier_percent = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 5:
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(df) * 100
            outlier_percent.append(outlier_pct)
        else:
            outlier_percent.append(0.0)

    summary_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Missing (%)': missing_percent,
        'Outlier (%)': outlier_percent
    }).sort_values('Missing (%)', ascending=False)

    plt.figure(figsize=(16, 8))
    x = np.arange(len(summary_df))
    plt.bar(x, summary_df['Missing (%)'], color='#3498db', alpha=0.8, label='Missing (%)')
    plt.bar(x, summary_df['Outlier (%)'], bottom=summary_df['Missing (%)'], 
            color='#e74c3c', alpha=0.7, label='Outlier (%)')
    plt.xticks(x, summary_df['Feature'], rotation=60, ha='right')
    plt.ylabel('Percentage')
    plt.title(f'Post-Imputation Quality - {timestamp_prefix}')
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{timestamp_prefix}_PostImpute_Quality_Bar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Post-imputation analysis saved in: {output_dir}")