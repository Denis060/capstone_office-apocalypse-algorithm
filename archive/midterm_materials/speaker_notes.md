Speaker Notes — 12-minute Presentation

Timing guide: total 12:00 minutes (720 seconds). Aim for ~10-11 content slides + intro & Q&A buffer.

0:00 - 0:30 (30s) — Slide 1 (Title)
- Quick team introduction (names & roles)
- One-line project elevator pitch: "We predict NYC office building vacancy risk by integrating property, transaction, mobility, permit and storefront datasets to produce building-level risk scores."

0:30 - 1:30 (60s) — Slide 2-3 (Introduction & Problem Statement)
- Explain why office vacancy matters (economic, social, tax base)
- State the goal clearly: early warning risk scoring for stakeholders

1:30 - 3:00 (90s) — Slides 4-7 (Background / Literature Review)
- Summarize prior approaches: hedonic models, time-series, supervised classification
- Explain gaps: few works integrate transaction + mobility + permit data at BBL level

3:00 - 4:30 (90s) — Slides 8-9 (Datasets)
- Briefly list the six datasets and their role (PLUTO, ACRIS, MTA, DOB, Business Registry, Storefronts)
- Emphasize BBL-based integration and temporal coverage (2020-2024 for mobility)

4:30 - 7:00 (150s) — Slides 10-12 (EDA & Key Visuals)
- Show the PLUTO missing-data figure and summarize (data quality, office area gaps)
- Show MTA ridership figure: point out Manhattan concentration and ridership distribution
- Show ACRIS transaction trends: annual volumes and borough distribution
- State 2–3 key EDA insights (e.g., ridership decline correlates with storefront vacancy)

7:00 - 9:00 (120s) — Slides 13-16 (Proposed Solution)
- Describe pipeline: ingestion -> feature engineering -> model -> explainability
- Model choices: XGBoost/LightGBM with SHAP, survival model for time-to-vacancy
- Highlight key engineered features (ACRIS frequency, MTA change vs baseline, permit counts)

9:00 - 10:00 (60s) — Slide 17 (Evaluation & Validation)
- Explain temporal split, geographic holdout, metrics (ROC-AUC, precision@k, C-index for survival)

10:00 - 11:00 (60s) — Slides 18-19 (Deployment & Timeline)
- Deliverables: risk table by BBL, dashboard, report
- Quick timeline to final deliverable and next steps

11:00 - 12:00 (60s) — Slide 20 (References) + Closing
- Quick mention of datasets and tools
- Invite questions; handoff to teammates for Q&A (divide speaking roles)

Tips for presenters
- Each team member speaks for a portion; rehearse transitions
- Keep each slide to 30–90 seconds depending on content density
- Have 2–3 backup slides with technical details in case of deep questions


Dataset slide speaking notes (Slide 8–9c) — 60–90s total
- Opening (10s): "We use six public NYC datasets to build our risk model — PLUTO, ACRIS, MTA ridership, DOB permits, Vacant Storefronts, and the Business Registry."
- Why these datasets (20s): "PLUTO is the canonical property master with BBL keys; ACRIS provides transaction-level financial activity; MTA provides mobility/foot-traffic signals; DOB shows investment/renovation activity; Storefronts provides on-the-ground vacancy labels; Business Registry gives business churn context."
- Size & handling (20s): "Data ranges from several hundred MB to multi-GB raw files. We aggregate (MTA) and chunk (DOB) to produce compact feature tables joined by BBL for modeling. All sources are public NYC Open Data and have precedent in urban analytics literature."
- Transition (10s): "Next, we'll show EDA visuals that demonstrate how these sources contribute predictive signals for vacancy risk."

EDA figure talking points (Slide 10–12, ~150s total)
- PLUTO (30s): "This figure shows missingness and distributions for office-critical columns. We use these to plan imputation and to confirm BBL coverage for joins. Key point: small missingness in office area is handled by imputation + missingness flags."
- MTA (25s): "Ridership trends reveal strong central-business-district concentration and notable declines post-2020 in select stations — these become temporal features (change vs baseline) in the model."
- ACRIS (25s): "Transaction frequency and sale price distributions indicate which buildings are actively traded or held; sudden drops in transaction frequency can be an early red flag for vacancy risk."
- Business Registry (20s): "We show borough counts and top reported types (small sample). Business density and concentration of certain license types (e.g., retail vs professional services) correlate with storefront vitality."
- DOB & Storefronts status (20s): "DOB permit aggregates and storefront monthly counts are important signals. The sample-generation script is available; if figures are missing during the presentation, mention that the figures can be regenerated locally (see README) — we have already produced most visuals and will include the DOB/storefront charts in the final report."


Appendix: Short answers for likely questions
- Label definition: storefront vacancy + business license inactivity -> building-level vacancy proxy
- Handling missing BBLs: geocoding or nearest-neighbor spatial join to PLUTO
- Ethical considerations: model is decision-support, not deterministic; include fairness checks
