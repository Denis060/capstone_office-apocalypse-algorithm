# Technical Paper Draft 2 - UPDATED OUTLINE
**Office Apocalypse Algorithm: Building-Level Vacancy Risk Prediction with Municipal Data Integration and Data Quality Validation**

## UPDATED Abstract (Matches Actual Work)

This paper presents a methodologically rigorous approach to predicting office building vacancy risk in NYC using integrated municipal datasets. Through comprehensive data quality validation, we identified and resolved critical data leakage issues that initially produced unrealistic 99%+ accuracy scores. Our final model—a calibrated logistic regression—achieves 88.2% ROC-AUC on 7,191 NYC office buildings, demonstrating the importance of conservative feature engineering for reliable real-world deployment. The methodology integrates six municipal data sources (PLUTO, ACRIS, DOB, MTA, Business Registry, Storefronts) using BBL-based joins and temporal validation frameworks to prevent future information leakage. Key contributions include: (1) a systematic approach to detecting and resolving data leakage in municipal datasets, (2) temporal validation strategies for building-level prediction, (3) calibrated probability outputs enabling policy prioritization, and (4) demonstration that realistic performance metrics (88%) provide more value than inflated scores (99%) for municipal applications.

## NEW Paper Structure (Reflects Actual Methodology)

### 1. Introduction
- NYC office vacancy post-pandemic challenge
- **Research question**: Reliable building-level prediction using municipal data integration
- Emphasis on methodology rigor vs performance maximization

### 2. Data Sources & Integration
- **Six municipal datasets**: PLUTO, ACRIS, DOB, MTA, Business Registry, Storefronts
- **BBL-based integration**: Building-level joins across heterogeneous sources  
- **Office building filtering**: 7,191 buildings from 857K total properties
- **Data quality challenges**: Identifying appropriate features for prediction

### 3. Methodology & Data Quality Discovery
- **Initial results**: 99.6% accuracy raised red flags
- **Data leakage investigation**: Target variable embedded in predictor features
- **Conservative feature removal**: Eliminated ALL derived/composite features
- **Clean feature set**: 20 raw building characteristics only
- **Temporal validation**: 4-strategy framework preventing future information leakage

### 4. Model Implementation
- **Logistic regression baseline**: L2 regularization with StandardScaler
- **Probability calibration**: CalibratedClassifierCV for reliable probability interpretation
- **Binary classification**: High-risk (30%) vs Low-risk (70%) buildings
- **Output format**: 0.0-1.0 probability scores for each building

### 5. Results & Validation
- **Clean performance**: 88.2% ROC-AUC, 81.7% accuracy
- **Business metrics**: 87.5% Precision@10% for policy targeting
- **Class distribution**: 2,157 high-risk vs 5,034 low-risk buildings
- **Probability examples**: Concrete building-level risk scores

### 6. Discussion & Policy Implications
- **Methodological rigor**: Data leakage detection as project strength
- **Realistic performance**: 88% trustworthy vs 99% artificial
- **Policy applications**: Building prioritization for intervention
- **Deployment readiness**: Conservative approach ensures reliability

### 7. Lessons Learned & Contributions
- **Data quality validation**: Critical for municipal ML applications
- **Conservative feature engineering**: Better safe than sorry
- **Temporal validation**: Essential for building-level prediction
- **Calibrated probabilities**: Policy decision-making requires interpretable outputs

## KEY UPDATES NEEDED

### Remove from Draft 1:
- ❌ XGBoost/SHAP claims (not implemented)
- ❌ 857K building analysis (wrong scale)
- ❌ Complex feature engineering (didn't do this)
- ❌ Theoretical frameworks (replace with actual methodology)

### Add to Draft 2:
- ✅ Data leakage discovery story (major contribution)
- ✅ 7,191 building focus (actual dataset)
- ✅ 88.2% ROC-AUC results (real performance)
- ✅ Temporal validation framework (actual methodology)
- ✅ Calibrated probability outputs (real deliverable)
- ✅ Conservative feature engineering (actual approach)

## TIMELINE FOR COMPLETION
- **This Week**: Draft new Results section with 88.2% ROC-AUC findings
- **Next Week**: Complete Methodology section with data leakage story
- **Following Week**: Finalize Discussion with policy implications
- **Final Week**: Polish and submit Technical Paper Draft 2

## Professor Meeting Response
**"You're absolutely right - our Draft 1 doesn't reflect our actual work. We've made significant methodological discoveries that are more valuable than what we originally proposed. We're working on Draft 2 that focuses on our data quality validation methodology and realistic 88% performance results rather than theoretical 99% claims."**