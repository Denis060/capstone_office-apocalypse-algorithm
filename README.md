# Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction

## Project Overview

The Office Apocalypse Algorithm is a comprehensive machine learning framework that predicts office building vacancy risk in New York City. This capstone project integrates six municipal datasets via Borough-Block-Lot (BBL) identifiers to generate building-level risk predictions for policy applications.

**Core Question**: Can we predict which NYC office buildings are at high risk of vacancy using integrated municipal data sources?

## Current Project Status (November 2025)

✅ **Data Integration Complete**: 7,191 NYC office buildings from 6 integrated datasets  
✅ **Data Leakage Resolved**: Identified and eliminated features causing 99%+ artificial accuracy  
✅ **Baseline Model Deployed**: Logistic regression achieving 88.2% ROC-AUC with clean data  
✅ **Temporal Validation**: 4-strategy framework preventing data leakage in predictions  
✅ **Professor Meeting Materials**: Comprehensive presentation package ready  

🔄 **In Progress**: Hyperparameter tuning and advanced model exploration  
📋 **Next Phase**: Final model evaluation, SHAP interpretation, technical paper completion

## Key Results

### Model Performance (Clean Dataset)
- **ROC-AUC**: 88.2% (excellent discrimination)
- **Accuracy**: 81.7% (realistic performance)  
- **Precision@10%**: 87.5% (if NYC targets top 10% riskiest buildings)
- **Dataset**: 2,157 high-risk vs 5,034 low-risk office buildings (30%/70% split)

### Technical Implementation  
- **Probability Scores**: Each building gets 0.0-1.0 risk probability
- **Calibrated Predictions**: CalibratedClassifierCV ensures reliable probability interpretation
- **Policy-Ready Outputs**: Buildings ranked by risk for intervention prioritization
- **Clean Features**: 20 variables from raw building characteristics only

## Project Structure

```
office_apocalypse_algorithm_project/
├── README.md                         # This file - project overview
├── requirements.txt                  # Python dependencies  
├── src/                             # Core source code
│   ├── temporal_validation.py        # Time-aware validation framework
│   ├── baseline_model.py             # Logistic regression with calibration
│   ├── advanced_models.py            # Random Forest & XGBoost models
│   ├── data_loader.py                # Data integration utilities
│   └── hyperparameter_tuning.py      # Model optimization
├── scripts/                         # Analysis and testing scripts  
│   ├── analyze_data_leakage.py       # Data quality investigation
│   ├── test_clean_models.py          # Model validation
│   ├── tune_and_shap.py              # Tuning + interpretability
│   └── create_presentation_visuals.py # Documentation graphics
├── docs/                            # Project documentation
│   ├── professor_presentation.md     # Meeting slides
│   ├── meeting_talking_points.md     # Presentation strategy  
│   ├── professor_questions.md        # Q&A preparation
│   └── visuals/                     # Documentation images
├── data/                            # Municipal datasets
│   ├── raw/                         # Original CSV files (6 sources)
│   ├── processed/                   # Cleaned integrated data
│   └── features/                    # Engineered variables
```

## Data Sources & Integration

### Core Datasets (6 NYC Municipal Sources)
1. **PLUTO**: Building characteristics, zoning, assessed values (857K records)
2. **ACRIS**: Property transactions, ownership changes (1.24M records)  
3. **MTA Ridership**: Transportation accessibility patterns (100M+ records)
4. **DOB Permits**: Construction activity, investment indicators (Multi-million records)
5. **Business Registry**: Economic vitality, business composition (66K records)
6. **Storefronts Vacancy**: Ground truth vacancy indicators (348K records)

### Integration Methodology
- **BBL-based joins**: Property-level integration with >95% match rates
- **Spatial analysis**: Proximity-based matching for transportation/business data
- **Temporal alignment**: Common reference periods across datasets
- **Quality validation**: Cross-dataset consistency checks

## Key Project Achievements

### Data Quality Discovery & Resolution
- **Identified Critical Data Leakage**: Found target variable embedded in predictor features
- **Methodological Rigor**: Caught suspicious 99%+ accuracy early, investigated systematically  
- **Conservative Feature Selection**: Removed ALL derived features to ensure clean predictions
- **Validation Framework**: Implemented temporal validation preventing future information leakage

### Technical Implementation
- **Probability-Based Classification**: Each building gets calibrated 0.0-1.0 risk score
- **Policy-Ready Outputs**: Buildings ranked for intervention prioritization
- **Interpretable Models**: Focus on explainability for government applications
- **Realistic Performance**: 88.2% ROC-AUC with trustworthy, deployment-ready results

## Getting Started

### Environment Setup
```bash
# Create virtual environment  
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Usage
```bash
# Run baseline model
python src/baseline_model.py

# Analyze data quality  
python scripts/analyze_data_leakage.py

# Test clean models
python scripts/test_clean_models.py
```

### View Documentation
- **Professor Meeting Materials**: `docs/professor_presentation.md`
- **Technical Details**: `docs/meeting_talking_points.md`  
- **Q&A Preparation**: `docs/professor_questions.md`

## Academic Context

This capstone project demonstrates:
- **Data Science Methodology**: Proper validation, leakage detection, conservative feature engineering
- **Policy Applications**: Realistic performance metrics for municipal decision-making
- **Technical Rigor**: Clean implementation following ML best practices
- **Business Impact**: 87.5% precision for top 10% building targeting

## Next Steps

1. **Hyperparameter Optimization**: Grid search for final model selection
2. **Advanced Models**: Random Forest & XGBoost exploration  
3. **SHAP Interpretation**: Feature importance analysis for policy insights
4. **Technical Paper Completion**: Final methodology documentation

---

**Contact**: [Your contact information]  
**Institution**: [Your university/program]  
**Last Updated**: November 2025
- Multi-dataset integration framework
- XGBoost ensemble with SHAP explainability

**📋 NEXT PHASES** (Future Work)
- Technical Paper Draft 2 with experimental results
- Model deployment and real-time scoring
- Stakeholder dashboard development
- Final capstone presentation

## Contact & Repository

- **Repository**: [capstone_office-apocalypse-algorithm](https://github.com/Denis060/capstone_office-apocalypse-algorithm)
- **Branch**: main
- **License**: Academic/Research Use

---

*This project represents a novel approach to urban analytics by demonstrating meaningful integration of heterogeneous municipal administrative datasets for building-level vacancy prediction with full explainability for policy decision-making.*