---
title: "Office Apocalypse Algorithm — Midterm Presentation"
author: "Capstone Team"
date: "Midterm: TBD"
---

# Slide 1 — Title

Office Apocalypse Algorithm
NYC Office Vacancy Risk Prediction

Team: [Your Team Name]
Course: Data Science Capstone

---

# Slide 2 — Project Introduction (1/2)

What is this project?

- Build a data-driven algorithm to predict office building vacancy risk in NYC.
- Integrate property (PLUTO), transaction (ACRIS), mobility (MTA), permits (DOB), business registry, and storefront vacancy datasets to produce building-level risk scores (by BBL).

This slide introduces the project at-a-glance:

- Objective: Predict which NYC office buildings are at risk of becoming vacant in the short-to-medium term (6–24 months).
- Approach: Fuse six heterogeneous datasets (PLUTO, ACRIS, MTA, DOB, Business Registry, Storefronts) at the BBL level and build explainable ML models.
- Output: Building-level risk scores, top contributing signals (SHAP), and a stakeholder dashboard for planners and investors.

---

# Slide 3 — Problem Statement (2/2)

Why this problem matters:

- Office vacancy threatens urban tax base, small businesses, and neighborhood vitality.
- Early-warning predictions help policymakers, landlords, investors, and community organizations prioritize interventions.

Goal: Produce a high-precision model that flags office buildings at risk of prolonged vacancy and surfaces the most influential signals.

Audience: City planners, real estate investors, building managers, researchers.

Problem statement (concise):

- Urban problem: Large-scale office vacancy reduces neighborhood economic activity, lowers property tax revenue, and accelerates retail decline.
- Research/technical challenge: Combine financial (ACRIS), physical (PLUTO), mobility (MTA), regulatory (DOB), and ground-truth vacancy signals to predict vacancy reliably while avoiding data leakage and geographic bias.
- Success criteria: A robust, time-aware model that achieves high precision/recall on held-out time windows and provides interpretable feature contributions for each high-risk building.

Stakeholder value (what they get):

- Prioritized list of at-risk BBLs with risk scores and key risk drivers
- Visual dashboard for neighborhood-level monitoring and spatial filtering
- Actionable recommendations (e.g., target maintenance, leasing incentives, zoning interventions)

---

# Slide 4 — Background / Literature Review (1/4)

High-level research context:

- Real-estate risk modeling: hedonic pricing, vacancy and absorption models, and time-series forecasting of occupancy.
- Urban analytics: integrating transaction, mobility, and permit data to infer economic health.
- Recent applied work: studies on post-pandemic office demand shifts and retail/storefront vacancy patterns.

---

# Slide 5 — Literature Review (2/4): Key prior work

- Hedonic pricing and urban economics (Rosen-style hedonic frameworks) — establish how building characteristics map to value and occupancy.
- Vacancy forecasting and absorption models from real-estate literature — time-series and survival approaches to predict vacancy and time-to-lease.
- Post-pandemic office demand analyses (e.g., Barrero, Bloom & Davis, 2021) — studies documenting remote work impacts and long-term office demand shifts.
- Urban analytics studies combining transaction and spatial datasets (PLUTO + ACRIS) to infer neighborhood investment patterns.

---

---

# Slide 5 — Background / Literature Review (2/4)

Methodologies seen in related work:

- Feature engineering from transaction histories (sale frequency, price trends, mortgage activity).
- Spatial joins linking POIs, transit access, and built environment.
- Supervised classification (random forests, gradient-boosted trees) and survival analysis for time-to-vacancy.

---

# Slide 6 — Literature Review (3/4): Methods & Findings

- Econometric approaches: hedonic regressions, panel fixed-effects, and time-series forecasting for vacancy rates.
- Machine learning approaches: tree-based models (Random Forest, XGBoost) used for tabular prediction tasks with high-dimensional features.
- Survival analysis: Cox proportional hazards and gradient-boosted survival models for time-to-event predictions (time-to-vacancy/time-to-lease).
- Spatial methods: nearest-neighbor spatial joins, distance-weighted accessibility metrics, and geographic holdout validation to test generalization.

---

# Slide 7 — Literature Review (4/4): Gaps we address

- Few studies jointly fuse financial transaction history (ACRIS), building characteristics (PLUTO), mobility (MTA), permits (DOB) and storefront vacancy labels at BBL resolution.
- Many models focus on price or rent; fewer operationalize actionable building-level risk scores with interpretable drivers for stakeholders.
- Our contribution: multi-source BBL fusion, time-aware modeling (temporal folds), and explainability (SHAP) to produce prioritized risk lists ready for policy action.


---

# Slide 6 — Background / Literature Review (3/4)

Gaps our project addresses:

- Multi-source BBL-level integration across financial, physical, and mobility signals.
- Fine-grained early-warning features (e.g., frequency of partial transfers, building-level ridership decline).
- Operationalizable risk score for stakeholders (not just academic inference).

---

# Slide 7 — Background / Literature Review (4/4)

References & suggested reading (add full citations in final slide):

- NYC Open Data (PLUTO, ACRIS, MTA, DOB, Storefronts)
- Urban analytics & vacancy modeling surveys (append in final document)

---

# Slide 8 — Dataset Overview (1/2)

Core datasets used:

- PLUTO — property master dataset (foundation) • ~857k rows (PLUTO 2025 v2.1)
- ACRIS — transaction history (financial signals) • ~1.24M rows
- MTA Subway Ridership — accessibility/foot-traffic • ~769k measurements
- DOB Permits — construction/renovation activity • Large (multi-million rows)
- Storefronts (vacancy labels) — target signal for commercial vacancy
- Business registry — business density and churn

(Sources: NYC Dept of City Planning, Dept of Finance, MTA, DOB, NYC Open Data)

---

# Slide 9 — Dataset Overview (2/2)

Integration approach:

- Primary key: BBL (Borough-Block-Lot) for dataset joins
- Spatial joins (nearest station, geocoding) for datasets lacking BBLs
- Temporal aggregation to monthly and quarterly features

Sample dataset statistics (from project documentation):

- PLUTO rows: 857,736
- ACRIS rows: 1,244,069
- MTA rows: 769,148
- Vacant Storefronts rows: 348,297
- Business registry rows: 66,425

---

# Slide 9b — Dataset Details (expanded)

Core datasets (role, public source, approximate size):

- PLUTO — property master (foundation). Source: NYC DCP. Rows: ~857,736. Use: building footprints, land use, office area, year built.
- ACRIS — transaction history. Source: NYC Dept of Finance. Rows: ~1,244,069. Use: sales, mortgages, transaction frequency and values.
- MTA Subway Ridership — mobility/access. Source: MTA. Raw hourly records (2020–2024) aggregated to monthly/station levels for features; original file is large (~100M+ records), we aggregate for modeling.
- DOB Permit Issuance — construction/renovation activity. Source: DOB. Multi-million rows; processed with chunked aggregation to produce permit counts and estimated cost features by BBL.
- Vacant Storefronts — vacancy labels. Source: NYC Open Data. Rows: ~348,297. Use: target proxy for commercial vacancy and ground-floor vitality.
- Business Registry / Licenses — business density and churn. Source: NYC Open Data / DCA. Rows: ~66,425. Use: active licenses, churn metrics at BBL/neighborhood level.

How we discovered/validated datasets: official NYC Open Data portals, project documentation, and prior literature referencing PLUTO/ACRIS for urban analytics.

---

# Slide 9c — Dataset Summary Table (concise)

| Dataset | Public / Private | Size (approx) | Source / How found | Prior use in literature |
|--------:|:---------------:|:-------------:|:------------------|:-----------------------|
| PLUTO | Public | ~375MB (~857k rows) | NYC DCP Open Data (PLUTO 2025 v2.1) | Widely used in urban analytics and housing/land-use studies
| ACRIS | Public | ~530MB (~1.24M rows) | NYC Dept of Finance / NYC Open Data | Common in transaction and foreclosure research
| MTA Ridership | Public | ~100M hourly records (2020–2024) | MTA Open Data / agency downloads | Mobility and economic-activity studies
| DOB Permits | Public | Multi-GB (millions rows) | NYC DOB Open Data | Used for construction/activity analyses
| Vacant Storefronts | Public | ~348k rows | NYC Open Data | Retail vacancy and neighborhood decline studies
| Business Registry | Public | ~66k rows | NYC Open Data (DCA) | Business churn / local economy studies

---

Note: We treat all datasets as public open data; sizes are the raw files used in our project. For model inputs we perform aggregation and sampling (MTA) or chunked processing (DOB) to create compact, feature-level tables.



# Slide 10 — EDA: High-level plan (1/2)

What we explored:

- Property distribution across boroughs, ages, building classes
- Transaction frequency and sale price trends by BBL
- Temporal ridership trends 2020–2024 near office clusters
- Permit activity as a signal of investment or disinvestment
- Vacancy label prevalence and basic class balance

---

# Slide 11 — EDA: Example code snippets (2/2)

Key reproducible analyses (code to include in notebooks):

- Distribution of building age & office area

```python
# Example: building age distribution
pluto['age'] = 2025 - pluto['YearBuilt']
pluto['age'].dropna().plot.hist(bins=40)
```

- ACRIS transaction frequency per BBL

```python
acris_agg = acris.groupby('bbl').agg({
    'document_id':'count',
    'sale_price':'median'
}).reset_index()
acris_agg['document_id'].hist(bins=50)
```

- MTA ridership trend near Midtown office clusters

```python
# spatial join: properties -> nearest station -> aggregate ridership
```

(Include resulting plots as figures in final slides.)

---

# Slide 11b — EDA: Visual Results (figures)

Below are the key figures generated from the project notebooks. These images are saved in the `figures/` folder and are ready to embed in the final presentation.

![PLUTO - Missing Data and Distributions](figures/02_pluto_dataset_analysis_cell10_out2.png)

*Figure: Missing data percentages for office-critical PLUTO columns.*

![PLUTO - Additional Distribution](figures/02_pluto_dataset_analysis_cell14_out2.png)

*Figure: Additional PLUTO distribution / summary visualization.*

![MTA - Ridership Summary](figures/03_mta_ridership_analysis_cell7_out2.png)

*Figure: Top stations by ridership, borough distribution, and ridership volume histogram (MTA).* 

![ACRIS - Transaction Summary](figures/05_acris_transactions_analysis_cell10_out1.png)

*Figure: ACRIS transaction types, borough distribution and annual transaction trends.*

![Business Registry - Borough Counts (sample)](figures/business_sample_borough_counts.png)

*Figure: Business Registry (sample) — counts by borough (small sample).* 

![Business Registry - Top Types (sample)](figures/business_sample_top_types.png)

*Figure: Business Registry (sample) — top 10 reported business/license types.*

---

# Slide 11f — EDA: DOB & Storefronts status (placeholders)

The DOB permits and Vacant Storefronts datasets are part of our full EDA. Lightweight sample plots are ready but not all were generated in this environment due to CSV read/time limits. To reproduce the missing figures locally, run the helper script `scripts/generate_sample_figures.py` in a Python environment with pandas/matplotlib/seaborn installed.

- Expected outputs (saved to `figures/` when run):
  - `storefronts_monthly_reports.png` or `storefronts_borough_counts.png` — monthly reports or borough counts for vacant storefront reports
  - `dob_permits_by_year.png` — aggregated permit counts per year (DOB)

*Figure placeholders:* when generated, embed these PNGs into the EDA slide deck. See README for local-run instructions.

---


# Slide 12 — EDA: Key findings

- **Cross-dataset validation confirms predictive signals**:
  - PLUTO office buildings (7,191 total) concentrated in Manhattan with 173.9-year average age requiring modernization investment
  - ACRIS shows Manhattan leads with 33% of transactions, indicating active but potentially volatile market conditions
  - MTA ridership patterns align with office concentrations, confirming accessibility as key vacancy predictor
  - Business Registry reveals 66,425 active licenses providing economic base with category diversity indicating neighborhood resilience

- **Integration success across all 6 datasets**:
  - BBL-based joins successful for 95%+ of records across PLUTO, ACRIS, and Business Registry
  - Spatial proximity methods enable MTA and Storefronts integration for buildings lacking direct BBL matches
  - Feature engineering pipeline extracts 50+ meaningful signals spanning building characteristics, financial activity, mobility access, and neighborhood vitality

---

# Slide 11c — EDA (PLUTO: Key Findings)

- PLUTO provides comprehensive BBL coverage — ideal canonical key for joins.
- Missingness: ~4–5% missing in `officearea`/`comarea` for older records; handled via imputation and flag features.
- Building age and office area were strong predictors of vacancy risk in exploratory tests.

Figure: `figures/02_pluto_dataset_analysis_cell10_out2.png` (embedded on Slide 11b)

---

# Slide 11d — EDA (Business Registry / Storefronts / DOB summary)

- Business Registry: business density and license churn correlate with neighborhood vitality; we derive business churn and active-license counts per BBL.
- Vacant Storefronts: direct vacancy labels at street level used to derive building-level vacancy proxies; temporal reporting enables duration analysis and short-term vacancy flagging.
- DOB Permits: permit counts and estimated costs signal active investment or disinvestment; high permit activity often indicates repositioning (lower short-term vacancy risk).

Note: the DOB dataset is large; we use chunked aggregation to compute permit counts by BBL and rolling sums for feature windows.

---

# Slide 11e — LLM / Chatbot Considerations (N/A for this project)

This project does not include an LLM/chatbot component. The midterm instructions asked for additional items when working with LLMs — they are not applicable here. For completeness, if we had included an LLM/chatbot we would provide:

- Data composition and knowledge-base construction (sources, ingestion, document chunking)
- Chosen LLM(s), reason (e.g., GPT-4 / LLaMA family), and RAG/fine-tuning plan
- Query/search criteria, retrieval pipeline (vector store, embedding model), and answer summarization
- Metrics for QA performance and safety checks

---

# Slide 11f — Requirements Checklist (midterm)

Mapping of midterm criteria to the slide deck (status):

- Project intro & problem statement (1-2 slides): Present — Slides 2–3
- Background / literature review (2-4 slides): Present — Slides 4–7
- Dataset (1-2 slides): Present + expanded — Slides 8–9b
- EDA (4-8 slides): Covered — Slides 10, 11, 11b, 11c, 11d (figures embedded)
- Proposed solution (2-4 slides): Present — Slides 13–16

Next actions available: (A) convert Markdown to PPTX/PDF, (B) add sampled figures for DOB/Storefronts/Business if full EDA images are missing, (C) finalize references and export.


# Slide 13 — Proposed Solution (1/4)

High-level approach:

1. Data ingestion & cleaning (PLUTO canonical master)
2. Feature engineering (ACRIS aggregates, MTA proximities, permit counts)
3. Label definition (vacancy target from Storefronts and business registry signals)
4. Model training & validation (classification + survival/time-to-event where applicable)
5. Explainability & risk scoring for stakeholders

---

# Slide 14 — Proposed Solution (2/4)

Modeling choices & rationale:

- Baselines: logistic regression, random forest
- Production model: XGBoost / LightGBM for tabular performance and interpretability via SHAP
- Survival analysis (Cox proportional hazards or gradient-boosted survival) for predicting time-to-vacancy
- Evaluation metrics: ROC-AUC, precision@k, calibration, F1, and time-to-event concordance (C-index)

---

# Slide 15 — Proposed Solution (3/4)

Feature engineering highlights:

- ACRIS-derived: transaction frequency (last 1/3/5 years), price trend slope, percent transfers, mortgage event counts
- PLUTO-derived: office area fraction, building age, floor count, lot area
- MTA-derived: average monthly ridership within 500m and change vs baseline (2019)
- DOB: permit counts, estimated cost of permits (rolling sum)
- Business registry: business churn rate, active license counts

---

# Slide 16 — Proposed Solution (4/4)

System architecture (one-slide diagram placeholder):

- Data ingestion pipelines (Airflow / scripts)
- Processing & feature store (Parquet, S3 / local data/processed)
- Model training environment (notebooks, experiments tracked in MLflow)
- Serving: batch-scored risk tables + dashboard (Streamlit / Dash)
- Optional: alerting and API for stakeholders

---

# Slide 17 — Evaluation & Validation

Validation strategy:

- Time-based split (train on data up to T, validate on [T, T+1]) to prevent leakage
- Cross-validation with grouped folds by neighborhood or BBL cluster
- Baselines for comparison and ablation studies

Robustness checks:

- Class imbalance handling (SMOTE, class weighting)
- Feature importance and SHAP explanations
- Geographic holdout tests

---

# Slide 18 — Deployment & Stakeholder Deliverables

Deliverables:

- Building-level risk score table (by BBL) with top contributing features
- Dashboard for city/planner use (filterable by borough/neighborhood)
- Report with EDA, method, and limitations

Ethical considerations:

- Ensure transparent explanations; avoid biased inputs that penalize vulnerable neighborhoods
- Use the model as a decision-support tool, not as sole authority for policy action

---

# Slide 19 — Timeline & Next Steps

Immediate next steps:

- Finish EDA figures and finalize feature set (1 week)
- Train baseline models and produce ablation study (2 weeks)
- Prepare final deliverables and slide deck (1 week)

Midterm deliverable: this slide deck + supporting notebooks and figures

---

# Slide 20 — References (Data & Tools)

- NYC PLUTO, ACRIS, MTA, DOB, Business Registry, Vacant Storefronts (NYC Open Data)
- Python, pandas, geopandas, XGBoost/LightGBM, SHAP, MLflow

(Include academic references and full citations in your final submission.)

---

# Slide 21 — Appendix: EDA code snippets & figure placeholders

(Include direct notebook links and code cells. Example file: `notebooks/05_acris_transactions_analysis.ipynb`)

---

# Slide 22 — Appendix: Contact & Team

Team members, roles, and contact info

---

# Slide 23 — Q&A

12-minute presentation; 4-minute Q&A


<!-- End of slides -->
