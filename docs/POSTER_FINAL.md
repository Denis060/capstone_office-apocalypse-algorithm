# Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction

**Ibrahim Denis Fofanah (Team Leader) • Bright Arowny Zaman • Jeevan Hemanth Yendluri**  
**Faculty Advisor: Dr. Krishna Bathula**  
**PACE University - Seidenberg School of Computer Science & Information Systems**

---

## 🟦 COLUMN 1: INTRODUCTION & BACKGROUND

### ABSTRACT

**Problem:**  
NYC faces record-high post-pandemic office vacancies, threatening property values and tax revenue. Traditional assessments are reactive and lack building-level prediction.

**Objective:**  
Develop a machine learning model that predicts office building vacancy risk using NYC Open Data.

**Innovation:**
- First building-level predictive approach using six NYC datasets
- Novel **data leakage detection framework**
- SHAP-powered interactive dashboard for explainable decisions

**Significance:**  
Model improves targeting efficiency **3.1×** and reduces intervention costs by **85%**.

---

### LITERATURE REVIEW

**Real Estate Modeling:**  
Hedonic pricing theory explains how property attributes affect value. Post-COVID shifts require predictive frameworks beyond aggregated statistics.

**Municipal Data Challenges:**  
PLUTO/ACRIS studies show value in open datasets, but gaps persist in **building-level** prediction due to inconsistent identifiers and temporal misalignment.

**Machine Learning:**  
Gradient boosting outperforms linear models for real estate analytics; SHAP improves interpretability for policy use.

**Research Gaps Filled:**
- Leakage-free temporal modeling
- Integration of six datasets at building resolution
- Explainable ML for targeted interventions

---

---

## 🟦 COLUMN 2: DATA & METHODOLOGY

### DATASET & PREPROCESSING

**Data Sources (7,191 NYC office buildings):**
- **PLUTO:** Property characteristics, assessments, building age
- **ACRIS:** Real estate transactions, deed transfers
- **DOB Permits:** Construction activity indicators
- **MTA Ridership:** Transportation accessibility proxies
- **Business Registry:** Commercial activity density
- **Storefront Vacancy:** Neighborhood economic health

**Preprocessing:**
- BBL standardization across datasets
- Temporal alignment ensuring causality
- Geospatial reconciliation
- 20 engineered features: physical, financial, market, contextual

---

### 📊 CHART 1: Borough Distribution
![Chart 1](../figures/poster_charts/chart1_borough_distribution.png)

---

### 📊 CHART 2: Data Sources
![Chart 2](../figures/poster_charts/chart2_data_sources.png)

---

### METHODOLOGY

**Leakage Detection:**
- Correlation screening (>95%)
- Temporal validation
- Causality checks
- Business logic review

**Modeling Pipeline:**
- Temporal splits: rolling, expanding, borough-aware
- Algorithms: Logistic Regression, Random Forest, **XGBoost**
- Grid search + 5-fold cross-validation

**Explainability:**  
SHAP for global/local interpretation; geographic visualizations; Streamlit dashboard deployment.

---

### 📊 CHART 3: System Architecture
![Chart 3](../figures/poster_charts/chart3_system_architecture.png)

---

---

## 🟦 COLUMN 3: RESULTS

### PERFORMANCE METRICS

---

### 📊 CHART 8: Metrics Dashboard
![Chart 8](../figures/poster_charts/chart8_metrics_dashboard.png)

**Key Results:**
- **ROC-AUC: 92.41%** (excellent discrimination)
- **Precision@10%: 93.01%** (targeting accuracy)
- **85% cost reduction** ($3.6M vs $5M)
- **3.1× efficiency gain**

---

### 📊 CHART 4: Model Comparison
![Chart 4](../figures/poster_charts/chart4_model_comparison.png)

---

### 📊 CHART 5: Feature Importance
![Chart 5](../figures/poster_charts/chart5_shap_importance.png)

**Top Predictors:**
1. **Building age** (1.406) - Older = higher risk
2. **Construction activity** (1.149) - Market indicator
3. **Office area** (0.776) - Size affects attractiveness
4. **Office ratio** (0.667) - Space efficiency

---

### GEOGRAPHIC INSIGHTS

**Borough Risk Distribution:**
- Brooklyn: **40.9%** (highest)
- Queens: 32.9%
- Bronx: 27.9%
- Staten Island: 25.5%
- Manhattan: 22.1% (lowest)

---

### 📊 CHART 6: Borough Risk
![Chart 6](../figures/poster_charts/chart6_borough_risk.png)

---

### BUSINESS IMPACT

**Random Targeting:**  
30% success → $5M cost

**Model-Driven:**  
93% success → $3.6M cost

**Result:** 85% cost reduction + 123% more interventions

---

### 📊 CHART 7: Business Impact
![Chart 7](../figures/poster_charts/chart7_business_impact.png)

---

---

## 🟦 COLUMN 4: CONCLUSIONS & CONTACT

### CONCLUSIONS

**Key Contributions:**
- Developed NYC's first building-level vacancy risk model
- Introduced systematic leakage detection framework
- Achieved **92.41% ROC-AUC** with high targeting precision
- SHAP interactive dashboard supports transparent policy decisions

**Research Questions Answered:**

**Q1:** Can ML predict vacancy risk?  
**A:** Yes, 92.41% ROC-AUC accuracy

**Q2:** Key drivers?  
**A:** Building age, construction activity

**Q3:** Practical deployment?  
**A:** Dashboard with geographic targeting

**Impact on Field:**
- **Data Science:** Robust leakage prevention framework
- **Urban Planning:** Evidence-based risk assessment
- **Real Estate:** Open data viability demonstrated

**Future Work:**
- Multi-city generalization
- Real-time economic indicators
- Causal feature engineering

---

### REFERENCES

1. Chen & Guestrin (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD*

2. Lundberg & Lee (2017). "A unified approach to interpreting model predictions." *NIPS*

3. NYC Dept. of Finance (2025). "Property Assessment Data (PLUTO)." *NYC Open Data*

4. Molnar (2022). *Interpretable Machine Learning*

---

### ACKNOWLEDGEMENTS

Special thanks to **Dr. Krishna Bathula** for guidance.  
Appreciation to PACE Seidenberg and NYC Open Data.

---

### CONTACT

**Team Lead:**  
Ibrahim Denis Fofanah  
if57774n@pace.edu

**Team Members:**  
Bright Arowny Zaman - bz75499n@pace.edu  
Jeevan Hemanth Yendluri - jy44272n@pace.edu

**GitHub:**  
github.com/Denis060/capstone_office-apocalypse-algorithm

**Live Demo:**  
Interactive Streamlit dashboard available

---

---

## 📋 CHART PLACEMENT GUIDE

| Chart | Location | Description |
|-------|----------|-------------|
| **1** | Column 2, Top | Borough Distribution (Pie) |
| **2** | Column 2, Middle | Data Sources (Bar) |
| **3** | Column 2, Bottom | System Architecture |
| **8** | Column 3, Top | Metrics Dashboard |
| **4** | Column 3, Upper-Mid | Model Comparison |
| **5** | Column 3, Middle | SHAP Importance |
| **6** | Column 3, Lower-Mid | Borough Risk |
| **7** | Column 3, Bottom | Business Impact |

---

**DESIGN NOTES:**
- Use **36-48pt** for section headers
- Use **24-28pt** for body text
- Use **20-22pt** for chart labels
- PACE Blue (#003C7D) for headers
- PACE Gold (#FFB81C) for highlights
- White background with 2" margins
- Print at 36" × 48" @ 300 DPI
