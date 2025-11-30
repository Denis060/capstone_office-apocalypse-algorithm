# Office Apocalypse Algorithm - Interactive Dashboard

🏆 **Champion Model: XGBoost (92.41% ROC-AUC)**

## Overview

Interactive Streamlit web application for NYC office building vacancy risk prediction. Built with our champion XGBoost model that achieved 92.41% ROC-AUC performance with clean features (no data leakage).

## Features

### 🏠 Building Lookup
- Individual building risk analysis
- SHAP-based prediction explanations  
- Building characteristics display
- Risk gauge visualization

### 📊 Risk Overview
- Portfolio-wide risk analysis
- Risk distribution visualization
- Top highest-risk buildings identification
- Summary statistics

### 🗺️ Risk Map
- Geographic risk visualization (if coordinates available)
- Borough-based risk analysis
- Interactive map with risk color coding

### 🎯 Intervention Planning
- Targeted intervention prioritization
- Customizable risk thresholds
- Building size filtering
- CSV export for action lists

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Dashboard:**
   ```bash
   streamlit run office_apocalypse_dashboard.py
   ```

3. **Open Browser:**
   Navigate to `http://localhost:8501`

## Data Requirements

The dashboard expects these files to exist:
- `../data/processed/office_buildings_clean.csv` - Clean building dataset
- `../models/champion_xgboost.pkl` - Trained XGBoost model
- `../models/champion_features.txt` - Feature names list

## Model Performance

- **Algorithm:** XGBoost with hyperparameter optimization
- **ROC-AUC:** 92.41%
- **Precision@10%:** 93.01%  
- **Features:** 20 clean building characteristics
- **Data:** 7,191 NYC office buildings
- **No Data Leakage:** Conservative feature engineering approach

## Business Value

- **Risk Identification:** Identify buildings at highest vacancy risk
- **Resource Allocation:** Prioritize interventions effectively
- **Policy Support:** Evidence-based decision making
- **Stakeholder Communication:** Clear, interpretable results

## Technical Details

- **Framework:** Streamlit for interactive web app
- **Visualizations:** Plotly for interactive charts
- **Explanations:** SHAP for model interpretability
- **Deployment Ready:** Production-grade code structure

---

**Professor Presentation Success ✅**  
Dashboard delivered as promised with champion XGBoost model and full SHAP interpretability.