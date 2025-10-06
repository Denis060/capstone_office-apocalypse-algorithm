# Processed Data Directory
**Office Apocalypse Algorithm - Processed Data**

## Purpose
This directory contains cleaned and processed datasets ready for feature engineering and modeling. All files focus specifically on the **7,191 office buildings** used in our analysis.

## Files

### `office_buildings_processed.csv`
- **Records**: 7,191 office buildings (NYC-wide)
- **Features**: 139 columns including engineered features from all 6 datasets
- **Size**: ~6.6 MB
- **Purpose**: Complete dataset ready for ML pipeline
- **Usage**: Input for model training notebooks

## Data Pipeline
```
Raw Data (6 datasets) → Feature Engineering → office_buildings_processed.csv → ML Models
```

## Quality Metrics
- **Completeness**: 95% after intelligent imputation
- **Geographic Coverage**: All 5 NYC boroughs
- **Temporal Coverage**: 2019-2024 (5-year analysis window)
- **Feature Quality**: 75 features selected for final modeling (after variance selection)

## Last Updated
October 6, 2025

---
*This directory previously contained large files with all 857K+ NYC buildings. Cleaned to focus only on office buildings for efficient processing.*