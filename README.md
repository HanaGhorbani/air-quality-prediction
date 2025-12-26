```markdown
# Air Quality Forecasting – Complete End-to-End Pipeline  
**CO(GT) Concentration Forecasting (24h & 168h ahead)**

A full-featured, production-ready pipeline that processes the famous **AirQualityUCI** dataset, generates realistic synthetic data, performs deep exploratory analysis, advanced preprocessing, time-series imputation quality evaluation, multi-horizon modeling (1-day & 1-week ahead), and finally exports trained models to **ONNX** for easy deployment.

---

### Key Features
- Realistic **synthetic data generation** with real-world missing/outlier patterns  
- Comprehensive **initial & post-imputation analysis** (missing vs outliers, co-occurrence matrix, correlation heatmaps)  
- Advanced **time-aware preprocessing** (cyclic encoding, lags, rolling stats, trends, ratios)  
- **Time-series interpolation** + back-testing to evaluate imputation quality  
- Training & comparison of **Linear Regression** and **Random Forest** models  
- Export models to **ONNX** format (ready for C++, JavaScript, mobile, edge, etc.)  
- Full visualization suite (12+ high-quality plots)  
- End-to-end testing script using ONNX Runtime  

---

### Project Structure
```bash
AIRQUALITY_PROJECT/
├── airquality_project/
│   ├── __pycache__/
│   ├── data_analyzer.py              # EDA on raw & cleaned datasets
│   ├── data_generator.py             # Synthetic data generation
│   ├── data_preprocessor.py          # Feature selection, cleaning, imputation, feature eng.
│   ├── imputation_quality_check.py   # Back-testing imputation quality
│   ├── main.py                       # MAIN PIPELINE (run this)
│   ├── model_training.py             # Train & evaluate models + export to ONNX
│   └── visualization.py              # All plotting & visualization utilities
├── data/
│   └── raw/
│       └── AirQualityUCI.csv         # Original UCI dataset (place here)
├── outputs/
│   ├── analyses/                     # Initial/post-imputation & imputation_quality reports
│   ├── cleaned_data/                 # Cleaned & engineered datasets + test sample
│   ├── figures/                      # All generated figures/plots
│   ├── model_results/                # CSV with model performance metrics
│   └── onnx_models/                  # Exported ONNX models (rf_day.onnx, rf_week.onnx, ...)
└── tests/
    ├── __pycache__/
    ├── predictor.py                  # ONNX inference helper
    ├── preprocess_for_predict.py     # Preprocessing for test (day/week models)
    └── test_main.py                  # Test ONNX models on test_data.csv

```

---

### One-Command Execution
```bash
python main.py
```

This single command runs **all 8 steps** sequentially:

| Step | Description |
|------|-------------|
| 1    | Generate 5,000 rows of realistic synthetic data |
| 2    | Deep EDA on raw real & synthetic datasets |
| 3–4  | Advanced preprocessing (feature selection, block cleaning, time interpolation) |
| 5    | Merge real + synthetic → final combined dataset |
| 6    | Post-imputation quality analysis |
| 7    | Imputation quality back-testing (MAE/RMSE per feature) |
| 8    | Train Linear & Random Forest models → export to ONNX + full visualization |

---

### Important Outputs
| Path | Content |
|------|--------|
| `outputs/cleaned_data/AirQuality_Combined_Clean.csv` | Final modeling dataset |
| `outputs/cleaned_data/test_data.csv` | 100-row sample for inference testing |
| `outputs/onnx_models/*.onnx` | Deployable models (rf_day.onnx, rf_week.onnx, etc.) |
| `outputs/figures/` | 12+ professional plots (tables, bars, scatters, residuals, time-series) |
| `outputs/model_results/model_evaluation_results.csv` | Full model comparison |

---

### Test Deployed Models (ONNX)
After running `main.py`, test the exported models on fresh data:

```bash
python test_main.py
```

This script:
- Preprocesses the test sample using the exact same selected features
- Runs inference with ONNX Runtime
- Saves predictions + performance plots

Outputs:
- `outputs/figures/test_day_ahead_predictions.png`
- `outputs/figures/test_week_ahead_predictions.png`

---

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            onnxruntime skl2onnx dataframe-image
```

> Note: `dataframe-image` is optional (used only for styled result tables).

---

### Dataset
Place the original dataset here:
```
data/raw/AirQualityUCI.csv
```
Download: https://archive.ics.uci.edu/dataset/360/air+quality

---

### Highlights
- Time-based interpolation with forward/backward fill  
- Lag features up to 1 week, rolling statistics, trend & ratio features  
- Robust outlier handling (IQR clipping)  
- Real imputation quality assessment via back-testing  
- Ready-to-deploy ONNX models  
- Fully reproducible & well-documented  

---

Enjoy clean air and accurate forecasts!

