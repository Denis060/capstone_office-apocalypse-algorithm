# Academic Poster Content - Office Apocalypse Algorithm
**PACE University - Seidenberg School of Computer Science & Information Systems**

---

## TITLE OF STUDY/RESEARCH
**Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Prediction**

## LIST OF AUTHORS AND CO-AUTHORS / FACULTY ADVISOR
**Authors:** Ibrahim Denis Fofanah (Team Leader), Bright Arowny Zaman, and Jeevan Hemanth Yendluri  
**Faculty Advisor:** Dr. Krishna Bathula  
**Institution:** PACE University - Seidenberg School of Computer Science & Information Systems

---

## ABSTRACT / INTRODUCTION

**Problem Statement:**
NYC faces unprecedented office building vacancy rates post-pandemic, creating economic and urban planning challenges. Traditional assessment methods are reactive rather than predictive.

**Research Objective:**
Develop a machine learning system to predict office building vacancy risk using publicly available data, enabling proactive intervention strategies.

**Key Innovation:**
- First systematic approach to office vacancy prediction using NYC open data
- Novel data leakage detection methodology ensuring realistic model performance
- Interactive dashboard for stakeholder decision-making

**Significance:**
Enables 3.1x more efficient resource allocation with 85% cost reduction compared to random targeting approaches.

---

## LITERATURE REVIEW / BACKGROUND

**Real Estate Risk Modeling:**
- Hedonic pricing theory (Rosen, 1974) establishes framework for building characteristics impact on market values
- Traditional models focus on aggregate indicators, limiting targeted intervention applicability
- Post-pandemic structural changes require new predictive approaches (Barrero et al., 2021)

**Municipal Data Integration Challenges:**
- Prior work with PLUTO/ACRIS datasets demonstrated value for neighborhood analysis (Furman Center, 2016)
- Multi-source integration hindered by inconsistent identifiers, temporal misalignment, scale differences
- Limited research operationalizes transportation accessibility and economic vitality indicators

**Machine Learning in Urban Analytics:**
- Gradient boosting methods show superior performance for real estate valuation (Yeh & Hsu, 2018)
- SHAP explainability enables policy-relevant interpretable predictions (Lundberg et al., 2020)
- Gap: Building-level prediction integrating heterogeneous municipal datasets remains underexplored

**Research Gaps Addressed:**
1. Systematic data leakage detection for temporal financial predictions
2. Integration of six municipal datasets at building resolution
3. Explainable ML deployment for policy intervention prioritization
4. Transportation demand patterns incorporated in vacancy risk models

---

## DATASET / DATA PREPROCESSING / EDA

**Data Sources (7,191 NYC Office Buildings):**
- **NYC PLUTO:** Property characteristics, assessments, building age
- **ACRIS:** Real estate transactions, deed transfers
- **DOB Permits:** Construction activity indicators
- **MTA Ridership:** Transportation accessibility proxies
- **Business Registry:** Commercial activity density
- **Storefront Vacancy:** Neighborhood economic health

**Data Integration Challenges:**
- **BBL (Borough-Block-Lot) Standardization:** Unified building identification across datasets
- **Temporal Alignment:** Ensuring feature-target temporal precedence
- **Geographic Coordinate Resolution:** Borough-level approximation for mapping

**Feature Engineering:**
```
Final Clean Features (20 total):
- Physical: building_age, lotarea, bldgarea, officearea, numfloors
- Financial: assessland, assesstot, value_per_sqft, land_value_ratio
- Market: transaction_count, deed_count, mortgage_count
- Contextual: mta_accessibility_proxy, business_density_proxy
- Risk: construction_activity_proxy, distress_score
```

**[CHART 1: Office Buildings Distribution by Borough]**
*Pie chart showing Manhattan (2,507), Brooklyn (1,776), Queens (1,619), Staten Island (705), Bronx (584)*

**[CHART 2: Data Sources Integration Overview]**
*Bar chart showing record counts from 6 municipal datasets*

---

## METHODOLOGY

**Data Leakage Detection Framework:**
1. **Correlation Analysis:** Identify features with >95% correlation to target
2. **Temporal Validation:** Test feature stability across time periods  
3. **Causality Verification:** Ensure features precede target measurement
4. **Domain Expert Review:** Validate business logic of feature relationships

**Model Development Pipeline:**
```
1. Temporal Validation Framework
   ├── Simple Temporal Split (80/20)
   ├── Rolling Window (6-month predictions)
   ├── Expanding Window (progressive training)
   └── Geographic Stratification (borough-aware)

2. Model Comparison
   ├── Logistic Regression (baseline)
   ├── Random Forest (ensemble)
   └── XGBoost (champion: 92.41% ROC-AUC)

3. Hyperparameter Optimization
   ├── Grid Search: 2.3 hours, 4-core Intel i7
   ├── 5-fold Stratified Cross-Validation
   └── Optimal: n_estimators=300, max_depth=6
```

**Interpretability Integration:**
- SHAP (SHapley Additive exPlanations) for feature importance
- Individual building risk factor analysis
- Geographic risk pattern visualization

**[CHART 3: System Architecture Diagram]**
*End-to-end pipeline: 6 Data Sources → ETL → Feature Engineering (20 features) → XGBoost Model → SHAP Analysis → Streamlit Dashboard*

---

## RESULTS AND ANALYSIS

**Champion Model Performance (XGBoost):**
| Metric | Value | Business Impact |
|--------|--------|-----------------|
| **ROC-AUC** | **92.41%** | Excellent discrimination capability |
| **Precision@10%** | **93.01%** | 93% accuracy targeting highest-risk buildings |
| **Precision@5%** | 95.12% | Exceptional accuracy for critical interventions |
| **F1-Score** | 0.847 | Balanced precision and recall |

**Model Comparison Results:**
```
Algorithm         ROC-AUC    Training Time    Business Value
XGBoost          92.41%     2.3 min         Champion Model
Random Forest    92.08%     1.8 min         Strong Baseline  
Logistic Reg.    88.20%     0.5 min         Interpretable Baseline
```

**[CHART 4: Model Performance Comparison]**
*Horizontal bar chart comparing ROC-AUC scores: XGBoost (92.41%), Random Forest (92.08%), Logistic Regression (88.20%)*

**Feature Importance (SHAP Analysis):**
| Rank | Feature | Impact | Interpretation |
|------|---------|--------|----------------|
| 1 | `building_age` | 1.406 | Older buildings (>50 years) exponentially higher risk |
| 2 | `construction_activity_proxy` | 1.149 | Market development activity predictor |
| 3 | `officearea` | 0.776 | Building size affects attractiveness |
| 4 | `office_ratio` | 0.667 | Space utilization efficiency |

**[CHART 5: SHAP Feature Importance Plot]**
*Waterfall plot showing top 5 features with building_age (1.406) dominating, followed by construction_activity_proxy (1.149)*

**Geographic Risk Analysis:**
```
Borough Risk Distribution:
Brooklyn    40.9% high-risk (highest)    1,776 buildings
Queens      32.9% high-risk             1,619 buildings  
Bronx       27.9% high-risk               584 buildings
Staten Is.  25.5% high-risk               705 buildings
Manhattan   22.1% high-risk (lowest)    2,507 buildings
```

**[CHART 6: NYC Borough Risk Heat Map]**
*Color-coded bar chart: Brooklyn (RED - 40.9%), Queens (ORANGE - 32.9%), Bronx (YELLOW - 27.9%), Staten Island (LIGHT GREEN - 25.5%), Manhattan (GREEN - 22.1%)*

**Business Value Quantification:**
- **Random Targeting:** 30% success rate, $5M for 300 interventions
- **Model Targeting:** 93% success rate, $3.6M for 669 interventions  
- **ROI Improvement:** 85% cost reduction + 123% more successful outcomes

**[CHART 7: Business Impact Visualization]**
*Side-by-side comparison bars showing Random (30% success, $5M cost) vs Model-Driven (93% success, $3.6M cost) with "85% Cost Reduction" annotation*

---

## CONCLUSIONS

**Key Contributions:**
1. **Methodological Innovation:** Systematic data leakage detection framework for financial prediction models
2. **Technical Achievement:** 92.41% ROC-AUC XGBoost classifier with 93% precision for highest-risk buildings
3. **Geographic Insights:** Brooklyn identified as highest-risk borough requiring targeted intervention
4. **Production Deployment:** Interactive Streamlit dashboard with SHAP explainability

**Research Questions Answered:**
- **Q1:** Can ML predict office vacancy risk? **A:** Yes, 92.41% ROC-AUC accuracy
- **Q2:** Which factors most influence risk? **A:** Building age dominates, followed by market context
- **Q3:** How to deploy practically? **A:** Interactive dashboard with geographic targeting

**Impact on Field:**
- **Data Science:** Robust framework for preventing data leakage in real estate analytics
- **Urban Planning:** Evidence-based risk assessment for proactive intervention
- **Real Estate:** Open data viability demonstrated for commercial property analysis

**Future Research Directions:**
- Multi-city generalization (Chicago, Boston, San Francisco)
- Real-time data integration with property management systems
- Causal feature engineering with economic indicators

---

## REFERENCES

1. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD*
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NIPS*
3. NYC Department of Finance. (2025). "Property Assessment Data (PLUTO)." *NYC Open Data Portal*
4. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*

---

## ACKNOWLEDGEMENTS OR CONTACT

**Special Thanks:**
- Dr. Krishna Bathula for research guidance and methodological insights
- Team collaboration: Ibrahim Denis Fofanah (Team Leader), Bright Arowny Zaman, Jeevan Hemanth Yendluri
- PACE University Seidenberg School for computational resources
- NYC Open Data initiative for comprehensive dataset access

**Contact Information:**
- **Team Lead:** Ibrahim Denis Fofanah {if57774n@pace.edu}
- **Team Members:** Bright Arowny Zaman {bz75499n@pace.edu}, Jeevan Hemanth Yendluri {jy44272n@pace.edu}
- **GitHub:** https://github.com/Denis060/capstone_office-apocalypse-algorithm
- **Dashboard Demo:** http://localhost:8501

**Interactive Demo Available:**
Live Streamlit dashboard with model predictions, SHAP explanations, and geographic risk visualization.

---

*This research demonstrates the intersection of machine learning, urban analytics, and practical policy applications, contributing to both academic knowledge and real-world problem-solving capabilities.*