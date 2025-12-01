# Academic Poster Content - Office Apocalypse Algorithm
**36" × 48" PACE University Poster Template Layout**

---

## TITLE (TOP CENTER - FULL WIDTH)
**Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction**

**Authors:** Ibrahim Denis Fofanah (Team Leader), Bright Arowny Zaman, Jeevan Hemanth Yendluri  
**Faculty Advisor:** Dr. Krishna Bathula  
**PACE University - Seidenberg School of Computer Science & Information Systems**

---

# 🟦 COLUMN 1 — ABSTRACT / INTRODUCTION + LITERATURE REVIEW
**(Text Only - No Charts)**

---

## ABSTRACT / INTRODUCTION

**Problem:**
NYC faces record-high post-pandemic office vacancies, threatening property values and tax revenue. Traditional assessments are reactive and lack building-level prediction.

**Objective:**
Develop a machine learning model that predicts office building vacancy risk using NYC Open Data.

**Innovation:**
• First building-level predictive approach using six NYC datasets
• Novel **data leakage detection framework**
• SHAP-powered interactive dashboard for explainable decisions

**Significance:**
Model improves targeting efficiency **3.1×** and reduces intervention costs by **85%**.

---

## LITERATURE REVIEW / BACKGROUND

**Real Estate Modeling:** Hedonic pricing theory explains how property attributes affect value. Post-COVID shifts require predictive frameworks beyond aggregated statistics.

**Municipal Data Challenges:** PLUTO/ACRIS studies show value in open datasets, but gaps persist in **building-level** prediction due to inconsistent identifiers and temporal misalignment.

**Machine Learning:** Gradient boosting outperforms linear models for real estate analytics; SHAP improves interpretability for policy use.

**Research Gaps Filled:**
• Leakage-free temporal modeling
• Integration of six datasets at building resolution
• Explainable ML for targeted interventions

---

---

# 🟦 COLUMN 2 — DATASET / PREPROCESSING / EDA + METHODOLOGY
**(Contains Charts 1, 2, and 3)**

---

## DATASET / DATA PREPROCESSING / EDA

**Data Sources (7,191 NYC office buildings):**
PLUTO (attributes), ACRIS (transactions), DOB Permits, MTA Ridership, Business Registry, Storefront Vacancy.

**Preprocessing Highlights:**
• BBL standardization
• Temporal alignment ensuring causality
• Geospatial reconciliation
• 20 engineered features across physical, financial, market & contextual categories

---

### 📊 **INSERT CHART 1 HERE**
**[Office Buildings Distribution by Borough — Pie Chart]**

---

### 📊 **INSERT CHART 2 HERE**
**[Data Sources Integration Overview — Bar Chart]**

---

## METHODOLOGY

**Leakage Detection:**
• Correlation screening (>95%)
• Temporal validation
• Causality checks
• Business logic review

**Modeling Pipeline:**
• Temporal splits: rolling, expanding, and borough-aware
• Algorithms tested: Logistic Regression, Random Forest, **XGBoost**
• Grid search + 5-fold cross-validation

**Explainability:**
SHAP for global and local interpretation; geographic visualizations; Streamlit dashboard deployment.

---

### 📊 **INSERT CHART 3 HERE**
**[System Architecture Diagram]**
*Place at bottom of Methodology section*

---

---

# 🟦 COLUMN 3 — RESULTS AND ANALYSIS
**(Contains Charts 4, 5, 6, and 7 — Bulk of visuals)**

---

## RESULTS AND ANALYSIS

**Champion Model (XGBoost):**
• ROC-AUC: **92.41%**
• Precision@10%: **93.01%**
• Precision@5%: **95.12%**
• F1-Score: **0.847**

**Model Comparison:**
• XGBoost: 92.41%
• Random Forest: 92.08%
• Logistic Regression: 88.20%

---

### 📊 **INSERT CHART 4 HERE**
**[Model Performance Comparison — Bar Chart]**

---

### 📊 **INSERT CHART 5 HERE**
**[SHAP Feature Importance Plot]**

---

**Geographic Risk Results:**

• Brooklyn: **40.9%** (highest)
• Queens: 32.9%
• Bronx: 27.9%
• Staten Island: 25.5%
• Manhattan: 22.1%

---

### 📊 **INSERT CHART 6 HERE**
**[Borough Risk Heatmap — Bar Chart]**

---

**Business Impact:**

• Random: 30% success → $5M
• Model-based: **93% success → $3.6M**
→ **85% lower cost + 123% more interventions**

---

### 📊 **INSERT CHART 7 HERE**
**[Business Impact Visualization]**

---

---

# 🟦 COLUMN 4 — CONCLUSIONS + REFERENCES + CONTACT
**(Text Only - No Charts)**

---

## CONCLUSIONS

**Key Contributions:**
• Developed NYC's first building-level vacancy risk model
• Introduced systematic leakage detection
• Achieved **92.41% ROC-AUC** with high targeting precision
• SHAP interactive dashboard supports transparent policy decisions

**Research Questions Answered:**

1. Can ML predict vacancy risk? → **Yes**
2. Key drivers? → **Building age**, construction activity
3. Practical deployment? → Dashboard with geographic targeting

**Future Work:**
• Expansion to other cities
• Real-time economic indicators
• Causal feature engineering

---

## REFERENCES

Chen & Guestrin (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD*

Lundberg & Lee (2017). "A unified approach to interpreting model predictions." *NIPS*

NYC Department of Finance (2025). "Property Assessment Data (PLUTO)." *NYC Open Data Portal*

Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*

---

## ACKNOWLEDGEMENTS / CONTACT

Special thanks to **Dr. Krishna Bathula**.
Appreciation to PACE Seidenberg and NYC Open Data.

**Team Lead:** Ibrahim Denis Fofanah – if57774n@pace.edu

**Team Members:**
Bright Arowny Zaman – bz75499n@pace.edu
Jeevan Hemanth Yendluri – jy44272n@pace.edu

**GitHub:** github.com/Denis060/capstone_office-apocalypse-algorithm

---

---

# ✅ **CHART PLACEMENT REFERENCE**

| Chart # | Title                         | Column   | Exact Placement       |
|---------|-------------------------------|----------|-----------------------|
| **1**   | Borough Distribution          | Column 2 | Under Dataset text    |
| **2**   | Data Integration Overview     | Column 2 | Under Chart 1         |
| **3**   | System Architecture           | Column 2 | Bottom of Methodology |
| **4**   | Model Performance Comparison  | Column 3 | Top of Results        |
| **5**   | SHAP Importance               | Column 3 | Under Chart 4         |
| **6**   | Borough Risk Heatmap          | Column 3 | Under SHAP chart      |
| **7**   | Business Impact Visualization | Column 3 | Bottom of column      |

**Note:** Chart 8 (Metrics Dashboard) is available as an alternative compact visual if needed.