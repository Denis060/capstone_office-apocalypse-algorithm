# Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](https://github.com/Denis060/capstone_office-apocalypse-algorithm)

## Project Overview

The Office Apocalypse Algorithm is a production-ready machine learning framework that predicts office building vacancy risk in New York City. This capstone project integrates six municipal datasets via Borough-Block-Lot (BBL) identifiers to generate building-level risk predictions for policy applications.

**Core Question**: Can we predict which NYC office buildings are at high risk of vacancy using integrated municipal data sources?

**Team**: Ibrahim Denis Fofanah (Leader), Bright Arowny Zaman, Jeevan Hemanth Yendluri  
**Advisor**: Dr. Krishna Bathula  
**Institution**: PACE University

---

## 🎯 Current Project Status (November 30, 2025)

### ✅ COMPLETE - Final Submission Ready

- ✅ **Champion Model Deployed**: XGBoost achieving **92.41% ROC-AUC** (vs 88.2% baseline)
- ✅ **Production Dashboard**: Streamlit app with SHAP interpretability deployed
- ✅ **IEEE Technical Paper**: Complete conference paper (LaTeX + Markdown)
- ✅ **Academic Poster**: Full content with 7 professional visualizations
- ✅ **Data Leakage Eliminated**: Systematic validation ensuring realistic performance
- ✅ **Comprehensive Documentation**: Professor-ready materials and submission package

---

## 🏆 Final Model Performance

### Champion Model: XGBoost Ensemble
- **ROC-AUC**: **92.41%** (excellent discrimination)
- **Accuracy**: 87.62% (realistic deployment performance)  
- **Precision@10%**: **93.01%** (93 of top 100 predictions correct)
- **Business Impact**: **3.1× efficiency** vs random targeting, **85% cost reduction**
- **Dataset**: 7,191 NYC office buildings (2,157 high-risk, 5,034 low-risk)

### Geographic Insights
- **Brooklyn**: 40.9% high-risk rate (highest vulnerability)
- **Manhattan**: 22.1% high-risk rate (most stable market)
- **Bronx**: 32.6% high-risk rate (emerging concerns)
- **Queens**: 28.4% high-risk rate (moderate risk)

### Technical Implementation  
- **XGBoost Ensemble**: Gradient boosting with 200 estimators, max depth 5
- **SHAP Interpretability**: Feature importance analysis for policy insights
- **Calibrated Probabilities**: Each building gets reliable 0.0-1.0 risk score
- **Production Dashboard**: Streamlit interface for interactive predictions
- **Clean Features**: 36 engineered variables from raw building/economic data

## Project Structure

```
office_apocalypse_algorithm_project/
├── README.md                              # Project overview & setup
├── SUBMISSION_CHECKLIST.md                # Final submission guide
├── requirements.txt                       # Python dependencies  
│
├── src/                                   # Core source code
│   ├── baseline_model.py                  # Logistic regression baseline
│   ├── advanced_models.py                 # XGBoost & Random Forest
│   ├── temporal_validation.py             # Time-aware validation
│   └── hyperparameter_tuning.py           # Model optimization
│
├── scripts/                               # Analysis & evaluation
│   ├── complete_evaluation.py             # Final model evaluation
│   ├── generate_poster_charts.py          # Academic poster visuals
│   ├── shap_model_interpretation.py       # SHAP analysis
│   ├── analyze_data_leakage.py            # Data quality checks
│   └── [other analysis scripts]
│
├── dashboard/                             # Production deployment
│   ├── app.py                             # Streamlit dashboard
│   ├── requirements.txt                   # Dashboard dependencies
│   └── README.md                          # Deployment instructions
│
├── docs/                                  # Complete documentation
│   ├── technical_paper_draft2.md          # Technical paper (Markdown)
│   ├── ieee_conference_paper_final.tex    # IEEE LaTeX paper
│   ├── academic_poster_content.md         # Poster content
│   ├── overleaf_latex_template.tex        # LaTeX template
│   └── [professor meeting materials]
│
├── models/                                # Trained models
│   ├── champion_xgboost.pkl               # Champion model (92.41% ROC-AUC)
│   └── champion_features.txt              # Feature list
│
├── results/                               # Evaluation outputs
│   ├── model_comparison.csv               # Algorithm comparison
│   ├── xgboost_shap_analysis.png          # SHAP visualizations
│   └── [other evaluation results]
│
├── figures/                               # Visualizations
│   └── poster_charts/                     # 7 academic poster charts
│
├── notebooks/                             # Jupyter analysis
│   └── 01-07_*.ipynb                      # EDA & dataset analyses
│
├── data/                                  # Municipal datasets (gitignored)
│   ├── raw/                               # Original CSV files (18GB)
│   ├── processed/                         # Cleaned data
│   └── features/                          # Engineered variables
│
└── archive/                               # Historical materials
    ├── deprecated_scripts/                # Old analysis code
    ├── draft_documents/                   # Earlier drafts
    └── midterm_materials/                 # Midterm presentation
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

## 🎓 Key Project Achievements

### 1. Champion Model Development
- **XGBoost Excellence**: 92.41% ROC-AUC outperforming baseline by 4.2 percentage points
- **Business Value**: 3.1× efficiency improvement, 85% operational cost reduction
- **High Precision**: 93.01% precision@10% for top building identification
- **Production Ready**: Deployed Streamlit dashboard with SHAP interpretability

### 2. Data Quality & Validation
- **Critical Leakage Detection**: Identified and eliminated features causing 99%+ artificial accuracy
- **Temporal Validation**: 4-strategy framework preventing future information leakage
- **Conservative Engineering**: Clean feature design ensuring realistic deployment performance
- **Cross-Validation**: Robust evaluation with stratified k-fold and geographic analysis

### 3. Academic Contributions
- **IEEE Conference Paper**: Complete technical paper with methodology, results, and conclusions
- **Academic Poster**: Professional 36"×48" poster with 7 visualizations
- **Comprehensive Documentation**: Professor-ready materials including LaTeX templates
- **Reproducible Research**: Clean code structure with full documentation

### 4. Policy-Ready Insights
- **Geographic Targeting**: Brooklyn identified as highest-risk borough (40.9% rate)
- **Feature Importance**: SHAP analysis revealing key vacancy drivers
- **Risk Scoring**: Calibrated probabilities enabling prioritized interventions
- **Operational Dashboard**: Interactive tool for building-level risk assessment

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip package manager
- Git (for cloning repository)

### Installation

```powershell
# Clone the repository
git clone https://github.com/Denis060/capstone_office-apocalypse-algorithm.git
cd capstone_office-apocalypse-algorithm

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```powershell
# Run complete model evaluation
python scripts/complete_evaluation.py

# Launch production dashboard
streamlit run dashboard/app.py

# Generate poster visualizations
python scripts/generate_poster_charts.py

# Analyze SHAP feature importance
python scripts/shap_model_interpretation.py
```

### 📊 View Key Deliverables

**Technical Documentation**
- IEEE Paper (LaTeX): `docs/ieee_conference_paper_final.tex`
- Technical Paper (Markdown): `docs/technical_paper_draft2.md`
- Academic Poster: `docs/academic_poster_content.md`

**Model & Results**
- Champion Model: `models/champion_xgboost.pkl`
- Evaluation Results: `results/model_comparison.csv`
- SHAP Analysis: `results/xgboost_shap_analysis.png`

**Visualizations**
- Poster Charts: `figures/poster_charts/` (7 professional charts)
- Dashboard: Run `streamlit run dashboard/app.py`

## 📈 Model Comparison

| Model | ROC-AUC | Accuracy | Precision@10% | F1-Score |
|-------|---------|----------|---------------|----------|
| **XGBoost (Champion)** | **92.41%** | **87.62%** | **93.01%** | **79.84%** |
| Random Forest | 91.67% | 86.89% | 91.25% | 78.12% |
| Logistic Regression | 88.22% | 81.73% | 87.50% | 72.34% |

**Business Impact**: Champion model achieves 3.1× efficiency vs random targeting with 85% cost reduction.

---

## 🏛️ Academic Context

### Capstone Project - Data Science Program
**Institution**: PACE University  
**Advisor**: Dr. Krishna Bathula  
**Team**: 
- Ibrahim Denis Fofanah (Team Leader)
- Bright Arowny Zaman
- Jeevan Hemanth Yendluri

### Project Demonstrates
- ✅ **Data Science Methodology**: Systematic leakage detection, temporal validation, rigorous evaluation
- ✅ **Machine Learning Engineering**: Production-ready models with SHAP interpretability
- ✅ **Policy Applications**: Geographic insights for NYC municipal decision-making
- ✅ **Technical Communication**: IEEE paper, academic poster, comprehensive documentation
- ✅ **Business Impact**: Quantified efficiency gains and cost reduction metrics

---

## 📚 Key Technologies

**Machine Learning**: XGBoost, scikit-learn, SHAP  
**Data Processing**: pandas, NumPy  
**Visualization**: matplotlib, seaborn, Plotly  
**Deployment**: Streamlit, pickle  
**Documentation**: LaTeX, Markdown, Overleaf

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{fofanah2025office,
  title={Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction},
  author={Fofanah, Ibrahim Denis and Zaman, Bright Arowny and Yendluri, Jeevan Hemanth},
  year={2025},
  institution={PACE University},
  note={Capstone Project - Data Science Program}
}
```

---

## 📧 Contact & Links

**GitHub Repository**: [https://github.com/Denis060/capstone_office-apocalypse-algorithm](https://github.com/Denis060/capstone_office-apocalypse-algorithm)  
**Project Lead**: Ibrahim Denis Fofanah  
**Last Updated**: November 30, 2025

---

## 📄 License

This project is developed for academic purposes as part of a capstone requirement at PACE University.

---

**Status**: ✅ **FINAL SUBMISSION COMPLETE** - Ready for professor review and final presentation

- **Repository**: [capstone_office-apocalypse-algorithm](https://github.com/Denis060/capstone_office-apocalypse-algorithm)
- **Branch**: main
- **License**: Academic/Research Use

---

*This project represents a novel approach to urban analytics by demonstrating meaningful integration of heterogeneous municipal administrative datasets for building-level vacancy prediction with full explainability for policy decision-making.*