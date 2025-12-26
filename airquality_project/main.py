# main.py
import os
import pandas as pd

from data_generator import generate_synthetic_data
from data_analyzer import analyze_dataset, analyze_clean_dataset
from data_preprocessor import (
    select_features,
    clean_missing_blocks,
    fill_missing_values,
    final_advanced_preprocessing,
    save_sample_for_testing
)
from imputation_quality_check import evaluate_imputation_quality
from model_training import train_and_evaluate

# =====================================================
# Paths
# =====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CLEANED_DIR = os.path.join(OUTPUTS_DIR, "cleaned_data")
ANALYSES_INITIAL_DIR = os.path.join(OUTPUTS_DIR, "analyses", "initial")
ANALYSES_POST_DIR = os.path.join(OUTPUTS_DIR, "analyses", "post_impute")
IMPUTATION_QUALITY_DIR = os.path.join(OUTPUTS_DIR, "analyses", "imputation_quality")

REAL_DATA_PATH = os.path.join(DATA_RAW_DIR, "AirQualityUCI.csv")
SYNTHETIC_DATA_PATH = os.path.join(CLEANED_DIR, "AirQuality_Synthetic_Generate.csv")

REAL_FINAL = os.path.join(CLEANED_DIR, "AirQuality_Final.csv")
SYNTHETIC_FINAL = os.path.join(CLEANED_DIR, "AirQuality_Synthetic_Final.csv")
COMBINED_CLEAN = os.path.join(CLEANED_DIR, "AirQuality_Combined_Clean.csv")
TEST_SAMPLE = os.path.join(CLEANED_DIR, "test_data.csv")

# =====================================================
# Make Dirs
# =====================================================
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(ANALYSES_INITIAL_DIR, exist_ok=True)
os.makedirs(ANALYSES_POST_DIR, exist_ok=True)
os.makedirs(IMPUTATION_QUALITY_DIR, exist_ok=True)

# =====================================================
# STEP 1: Generate synthetic data
# =====================================================
print("\nSTEP 1: Generating synthetic data...")
generate_synthetic_data(SYNTHETIC_DATA_PATH)

# =====================================================
# STEP 2: Initial analysis of raw datasets
# =====================================================
print("\nSTEP 2: Initial analysis of raw datasets...")

analyze_dataset(
    REAL_DATA_PATH,
    output_dir=ANALYSES_INITIAL_DIR,
    timestamp_prefix="Real_Raw"
)

analyze_dataset(
    SYNTHETIC_DATA_PATH,
    output_dir=ANALYSES_INITIAL_DIR,
    timestamp_prefix="Synthetic_Raw"
)

# =====================================================
# STEP 3: Preprocessing real data
# =====================================================
print("\nSTEP 3: Preprocessing real data...")
real_temp = os.path.join(CLEANED_DIR, "temp_real_selected.csv")

select_features(REAL_DATA_PATH, real_temp)
clean_missing_blocks(real_temp)
fill_missing_values(real_temp)
final_advanced_preprocessing(real_temp, REAL_FINAL)
save_sample_for_testing(REAL_FINAL, TEST_SAMPLE, n_rows=100, method="first")

if os.path.exists(real_temp):
    os.remove(real_temp)
    print(f"Temporary file removed: {real_temp}")

# =====================================================
# STEP 4: Preprocessing synthetic data
# =====================================================
print("\nSTEP 4: Preprocessing synthetic data...")
synth_temp = os.path.join(CLEANED_DIR, "temp_synth_selected.csv")

select_features(SYNTHETIC_DATA_PATH, synth_temp)
clean_missing_blocks(synth_temp)
final_advanced_preprocessing(synth_temp, SYNTHETIC_FINAL)

if os.path.exists(synth_temp):
    os.remove(synth_temp)
    print(f"Temporary file removed: {synth_temp}")

# =====================================================
# STEP 5: Create combined dataset after imputation
# =====================================================
print("\nSTEP 5: Creating combined dataset...")
df_real = pd.read_csv(REAL_FINAL, sep=';').assign(source='real')
df_synth = pd.read_csv(SYNTHETIC_FINAL, sep=';').assign(source='synthetic')

common_cols = df_real.columns.intersection(df_synth.columns).tolist()
df_combined = pd.concat([df_real[common_cols], df_synth[common_cols]], ignore_index=True)

df_combined.to_csv(COMBINED_CLEAN, sep=';', index=False)
print(f"Combined dataset saved: {COMBINED_CLEAN}")

# =====================================================
# STEP 6: Post-imputation analysis (combined)
# =====================================================
print("\nSTEP 6: Post-imputation analysis...")

analyze_clean_dataset(
    COMBINED_CLEAN,
    output_dir=ANALYSES_POST_DIR,
    timestamp_prefix="Combined_PostImpute"
)

# =====================================================
# STEP 7: Imputation quality evaluation
# =====================================================
print("\nSTEP 7: Evaluating imputation quality...")
try:
    evaluate_imputation_quality(
        clean_path=REAL_FINAL,
        raw_path=REAL_DATA_PATH,
        dataset_name="Real"
    )
    print("Imputation quality evaluation completed.")
except Exception as e:
    print("Error in imputation quality evaluation:", e)

# =====================================================
# Final Message
# =====================================================
print("\n" + "=" * 80)
print("All preprocessing and analysis steps were completed!")
print("Final outputs locations:")
print(f" • Raw data:                  {REAL_DATA_PATH}")
print(f" • Synthetic generated:       {SYNTHETIC_DATA_PATH}")
print(f" • Final real processed:      {REAL_FINAL}")
print(f" • Final synthetic processed: {SYNTHETIC_FINAL}")
print(f" • Combined for modeling:     {COMBINED_CLEAN}")
print(f" • Test sample:               {TEST_SAMPLE}")
print("=" * 80)

# =====================================================
# STEP 8: Model training & evaluation
# =====================================================
print("\nSTEP 8: Model training and evaluation...")
train_and_evaluate(
    data_path=COMBINED_CLEAN,
    outlier_method='clip',
    iqr_multiplier=1.5
)

print("\nAll done!")
print(f"• Figures → {os.path.join(OUTPUTS_DIR, 'figures')}")
print(f"• ONNX models → {os.path.join(OUTPUTS_DIR, 'onnx_models')}")
print(f"• Evaluation results → {os.path.join(OUTPUTS_DIR, 'model_results')}")