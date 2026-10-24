# Office Apocalypse Algorithm — Predicting NYC Office Vacancy Risk

> MS Data Science capstone (Pace University): a machine-learning early-warning system that ranks NYC office buildings by vacancy risk by fusing six public city datasets — with explicit data-leakage auditing and SHAP explainability.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Post-pandemic office demand left many NYC buildings at risk of prolonged vacancy. This project builds a building-level dataset of **7,191 office buildings** and trains classifiers to flag high-risk properties, then serves the results in an interactive Streamlit dashboard.

## Problem / Motivation

City agencies, lenders and owners need to know *which* buildings are likely to struggle before vacancy shows up in tax or lease data. Public data contains strong signals — building age and size, transit ridership, nearby business activity, sales/financing, permits, and storefront vacancy — but it is scattered across agencies and needs careful joining.

## Approach & tech stack

1. **Data integration** — six NYC Open Data sources, one notebook each: PLUTO (building attributes), MTA ridership, business registry, ACRIS property transactions, DOB permits, storefront vacancy.
2. **Feature engineering** — building, neighborhood and market features (e.g. building age, office-size, mixed-use, neighborhood vitality / vacancy-risk indices).
3. **Leakage audit** — `scripts/analyze_data_leakage.py` found that an early feature set produced near-perfect scores because some features encoded the target; the models were retrained on a cleaned feature set.
4. **Modeling** — Logistic Regression, Random Forest and XGBoost, 80/20 split (5,752 train / 1,439 test), hyperparameter tuning, robustness checks and dataset-ablation analysis.
5. **Explainability** — SHAP feature attributions per model.
6. **Delivery** — Streamlit dashboard (`dashboard/office_apocalypse_dashboard.py`), poster and presentation visuals.

**Stack:** Python · pandas · scikit-learn · XGBoost · SHAP · Plotly · Streamlit · NYC Open Data

## Key features

- Multi-source geospatial data fusion at the building (BBL) level
- Documented leakage detection and clean re-evaluation
- Dataset ablation study showing what each source contributes (`results/dataset_validation/`)
- Explainable risk scores via SHAP
- Interactive dashboard for exploring at-risk buildings

## Results (clean feature set, held-out test set)

From `results/clean_model_comparison.csv`:

| Model | Accuracy | ROC-AUC | Precision@10% | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.817 | 0.882 | 0.875 | 0.815 |
| Random Forest | 0.842 | 0.920 | 0.907 | 0.840 |

Additional XGBoost, tuning and robustness results are in `results/` and `models/`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/final_evaluation_clean.py               # retrain/evaluate on clean features
python scripts/shap_model_interpretation.py            # SHAP explanations
streamlit run dashboard/office_apocalypse_dashboard.py # dashboard
```

Raw NYC datasets are large and not all are committed; see the data notebooks in `notebooks/` for sources.

---

## About the author

**Ibrahim Denis Fofanah** — Data Scientist · ML / GenAI (RAG, agents) · BI & analytics · MS Data Science, Pace University (2025)

[LinkedIn](https://www.linkedin.com/in/ibrahim-denis-fofanah/) · [Email](mailto:ibrahim.fofanah.ds@gmail.com) · [Live RAG portfolio project](https://moijey-copilot.vercel.app) · [GitHub](https://github.com/Denis060)
