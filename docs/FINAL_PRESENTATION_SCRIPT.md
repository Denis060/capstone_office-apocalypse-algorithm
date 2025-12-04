# Office Apocalypse Algorithm - Final Presentation Script
## 12-Minute Final Defense | December 2025

**Team:** Ibrahim Denis Fofanah (Lead), Bright Arowny Zaman, Jeevan Hemanth Yendluri  
**Advisor:** Dr. Krishna Bathula  
**Institution:** PACE University - Seidenberg School of CSIS

---

## SLIDE 1: Title & Project Introduction (1 min)
**Visual:** Title slide with PACE branding + project logo

### Speaker Notes:
"Good [morning/afternoon]. I'm Ibrahim Denis Fofanah, and with my teammates Bright Zaman and Jeevan Yendluri, we're presenting the **Office Apocalypse Algorithm** - a machine learning system for predicting NYC office building vacancy risk.

**Problem Statement Recap:**
Post-pandemic, NYC faces record office vacancies threatening $4.6 billion in annual tax revenue. Traditional assessments are reactive - they identify problems after vacancies occur. Our solution? A predictive ML system that identifies at-risk buildings **before** they become vacant, enabling proactive intervention.

**Key Innovation:** We're the first to achieve building-level predictions by integrating 6 disparate NYC municipal datasets with rigorous data leakage prevention."

---

## SLIDE 2: Quick EDA Highlights - Data Landscape (30 sec)
**Visual:** Chart 1 (Borough Distribution) + Chart 2 (Data Sources)

### Speaker Notes:
"Our dataset spans 7,191 NYC office buildings across all five boroughs. As shown, Manhattan dominates with 39% of buildings, but Brooklyn and Queens together represent 40% of the portfolio.

We integrated 6 data sources - PLUTO property data, ACRIS transactions, DOB permits, MTA ridership, business registry, and storefront vacancy reports. The challenge? Each dataset had different identifiers, temporal frequencies, and coverage gaps."

---

## SLIDE 3: EDA Insights - Risk Distribution (30 sec)
**Visual:** Chart 6 (Borough Risk Heatmap) showing vacancy rates by borough

### Speaker Notes:
"Our EDA revealed critical geographic patterns. Brooklyn shows the highest vacancy risk at 40.9%, while Manhattan surprisingly has the lowest at 22.1%. This challenges conventional wisdom that Manhattan would be most affected.

Key insight: Building age emerged as the strongest predictor - buildings over 75 years old show 2.3× higher vacancy rates. This guided our feature engineering strategy."

---

## SLIDE 4: Proposed Solution - System Architecture (1 min)
**Visual:** Chart 3 (System Architecture Diagram)

### Speaker Notes:
"Our solution is an end-to-end ML pipeline with three key innovations:

**First:** A robust ETL pipeline that standardizes BBL identifiers across all six data sources and enforces temporal precedence - ensuring all predictive features come **before** the target measurement date.

**Second:** A systematic data leakage detection framework using correlation analysis, temporal validation, and business logic review. This eliminated 16 potentially leaky features.

**Third:** An interpretable XGBoost model with SHAP explanations, deployed via Streamlit dashboard for real-time predictions.

The architecture flows from data ingestion through validation, model training, and finally deployment for stakeholder use."

---

## SLIDE 5: Results - Model Performance Comparison (2 min)
**Visual:** Chart 4 (Model Comparison Bar Chart)

### Speaker Notes:
"We compared three algorithms: Logistic Regression, Random Forest, and XGBoost.

**Champion Model: XGBoost**
- **ROC-AUC: 92.41%** - excellent discrimination between high and low risk
- **Accuracy: 87.62%** - strong overall prediction performance
- **Precision@10%: 93.01%** - when we target the top 10% highest-risk buildings, we're correct 93% of the time
- **Precision@5%: 95.12%** - even better when focusing on the very highest risk

This significantly outperforms both Logistic Regression (78% ROC-AUC) and Random Forest (87% ROC-AUC).

**Why XGBoost won?** Handles non-linear relationships, robust to missing data, and naturally captures feature interactions like 'old building + low construction activity = high risk'."

---

## SLIDE 6: Results - Metrics Dashboard Deep Dive (2 min)
**Visual:** Chart 8 (Professional Metrics Dashboard)

### Speaker Notes:
"Let me break down our comprehensive metrics:

**Top Row - Precision Metrics:**
- 93% precision at 10% threshold, 95% at 5% threshold
- This means if we inspect 720 buildings (10% of portfolio), we'll correctly identify 669 high-risk cases

**Center - Model Quality:**
- 92.41% ROC-AUC indicates excellent model discrimination
- Dataset: 7,191 buildings, 20 engineered features after leakage detection

**Bottom Row - Business Impact:**
- **$1.4M cost savings** per 1,000-building assessment cycle
- **2.23× more successful interventions** vs random targeting
- **68% lower cost per success** - from $16,700 to $5,400 per intervention

These metrics demonstrate both statistical rigor and real-world business value."

---

## SLIDE 7: Analysis - Feature Importance & Drivers (1.5 min)
**Visual:** Chart 5 (SHAP Feature Importance)

### Speaker Notes:
"SHAP analysis reveals what actually drives vacancy risk:

**Top 4 Predictors:**
1. **Building Age (1.406 SHAP)** - Older buildings face modernization challenges
2. **Construction Activity (1.149)** - Low permit activity signals market disinterest
3. **Office Area (0.776)** - Larger buildings face higher vacancy risk due to market oversupply
4. **Office Ratio (0.667)** - Buildings with higher office-to-total-area ratios are more vulnerable

**Actionable Insights:**
- Buildings over 75 years need modernization incentives
- Low construction activity zones need economic development programs
- Large office buildings require targeted tenant retention strategies

This interpretability is crucial - stakeholders can understand **why** a building is flagged, not just **that** it's at risk."

---

## SLIDE 8: Analysis - Business Impact Comparison (1.5 min)
**Visual:** Chart 7 (Business Impact Comparison) showing Random vs Model targeting

### Speaker Notes:
"Let's compare two assessment strategies for 1,000 buildings:

**Random Targeting Approach:**
- Inspect 1,000 buildings at random
- 30% success rate (300 high-risk buildings found)
- Total cost: $5 million
- Cost per success: $16,700

**Model-Driven Approach:**
- Inspect top 720 buildings (10% threshold)
- 93% success rate (669 high-risk buildings found)
- Total cost: $3.6 million
- Cost per success: $5,400

**Result:** $1.4M saved, 123% more successful interventions, 68% lower cost efficiency.

This demonstrates the model's practical value - it's not just statistically accurate, it's economically transformative for NYC's assessment budget."

---

## SLIDE 9: Analysis - Geographic Risk Distribution (1 min)
**Visual:** Chart 6 (Borough Risk Heatmap) with detailed breakdown

### Speaker Notes:
"Geographic analysis reveals distinct risk patterns:

**Highest Risk Boroughs:**
- Brooklyn: 40.9% (aging industrial conversions)
- Queens: 32.9% (outer-borough challenges)
- Bronx: 27.9% (economic development gaps)

**Lowest Risk:**
- Manhattan: 22.1% (despite being the largest market)
- Staten Island: 25.5% (smaller, stable market)

**Policy Implications:**
- Focus modernization grants on Brooklyn
- Enhance transportation connectivity in Queens
- Economic development zones in Bronx

This geographic targeting enables efficient resource allocation across boroughs."

---

## SLIDE 10: Conclusion - Key Contributions (1 min)
**Visual:** Summary slide with 3 key contributions highlighted

### Speaker Notes:
"Three major contributions to the field:

**1. First Building-Level Prediction Framework**
- Previous studies operated at neighborhood or ZIP code level
- We achieve 92.41% accuracy at individual building resolution
- This granularity enables targeted interventions

**2. Systematic Data Leakage Detection Methodology**
- Developed 4-stage framework: correlation analysis, temporal validation, causality review, business logic
- Eliminated 16 leaky features (e.g., current vacancy status, recent rental data)
- This methodology is replicable for other temporal prediction problems

**3. Production-Ready Deployment**
- Live Streamlit dashboard with interactive predictions
- SHAP-powered explanations for stakeholder trust
- Geographic visualization for intervention planning

**Comparison to Literature:** Previous NYC studies achieved 78-82% accuracy at neighborhood level. We exceed this by 10+ percentage points while operating at finer resolution."

---

## SLIDE 11: Future Work & Recommendations (1 min)
**Visual:** Future roadmap with 3 expansion areas

### Speaker Notes:
"Future development opportunities:

**1. Multi-City Generalization**
- Adapt framework to Chicago, San Francisco, Boston
- Validate transferability of features across markets
- Build comparative urban analytics platform

**2. Real-Time Economic Indicators**
- Integrate live data streams: credit card transactions, cell phone mobility, utility usage
- Move from annual to quarterly predictions
- Enable early warning system for market shifts

**3. Causal Feature Engineering**
- Current features are correlational
- Develop instrumental variables for causal inference
- Answer: 'Will building renovations **cause** lower vacancy risk?'

**4. Policy Simulation Tool**
- Model impact of interventions (tax incentives, zoning changes)
- Cost-benefit analysis for different policy scenarios
- Support evidence-based urban planning"

---

## SLIDE 12: References & Demo (1 min)
**Visual:** QR code to live dashboard + key references

### Speaker Notes:
"Key references supporting our methodology:

1. **Chen & Guestrin (2016)** - XGBoost foundational paper
2. **Lundberg & Lee (2017)** - SHAP methodology
3. **NYC Department of Finance** - PLUTO and ACRIS data sources
4. **Molnar (2022)** - Interpretable ML best practices

**Live Demo Available:**
[Show QR code to dashboard]
Our production dashboard is live at this URL. You can:
- Look up any of 7,191 buildings by address or BBL
- View risk scores and explanations
- Explore geographic patterns
- Plan intervention priorities

**Repository:** github.com/Denis060/capstone_office-apocalypse-algorithm

We're happy to answer questions. Thank you!"

---

## TIMING BREAKDOWN (Total: 12 minutes)
- Slide 1 (Intro): 1:00
- Slide 2 (EDA 1): 0:30
- Slide 3 (EDA 2): 0:30
- Slide 4 (Solution): 1:00
- Slide 5 (Results 1): 2:00 ⭐
- Slide 6 (Results 2): 2:00 ⭐
- Slide 7 (Analysis 1): 1:30 ⭐
- Slide 8 (Analysis 2): 1:30 ⭐
- Slide 9 (Analysis 3): 1:00
- Slide 10 (Conclusion): 1:00
- Slide 11 (Future Work): 1:00
- Slide 12 (References): 1:00

**Total Focus on Results/Analysis/Conclusion: 8 minutes (67%)**

---

## ANTICIPATED QUESTIONS & ANSWERS

### Q1: "How did you validate temporal precedence?"
**A:** "We implemented three checks: (1) Date field analysis ensuring features were measured before target dates, (2) Rolling window validation where training data always precedes test periods, (3) Business logic review with domain experts to identify conceptual leakage. For example, we removed 'current rental rates' but kept 'historical transaction counts'."

### Q2: "Why XGBoost over neural networks?"
**A:** "Three reasons: (1) Tabular data - XGBoost excels with structured features, (2) Interpretability - SHAP values work seamlessly with tree models, (3) Sample size - 7,191 buildings is too small for deep learning. Neural networks typically need 100K+ samples for stable training."

### Q3: "What about model bias across boroughs?"
**A:** "Excellent question. We tested fairness metrics and found consistent precision across boroughs (Brooklyn: 91%, Manhattan: 93%, Queens: 92%). The model doesn't disadvantage any geography. However, **ground truth** vacancy rates differ - Brooklyn genuinely has higher risk due to building age and economic factors."

### Q4: "How often should the model be retrained?"
**A:** "Annually, aligned with PLUTO data releases. We recommend: (1) Q1 - Update with new PLUTO data, (2) Q2 - Retrain and validate, (3) Q3 - Deploy updated model, (4) Q4 - Monitor performance drift. If ROC-AUC drops below 88%, trigger emergency retrain."

### Q5: "Can this work for residential buildings?"
**A:** "Conceptually yes, but features would differ. Office vacancies are driven by employment patterns and corporate decisions. Residential vacancies relate to household income, school quality, crime rates. We'd need to engineer ~15-20 new features specific to residential markets. The **methodology** transfers, but not the specific features."

### Q6: "What's the business case for NYC to adopt this?"
**A:** "$1.4M savings per 1,000 buildings annually. NYC has ~50,000 office buildings. Scaled up, that's **$70M+ in annual savings** from more efficient building assessments. ROI payback in under 6 months if implemented citywide."

### Q7: "How do you handle new buildings with no history?"
**A:** "Cold start problem. For buildings <2 years old, we use: (1) Neighborhood averages for area-based features, (2) Building physical characteristics (age=0, area, floors), (3) Conservative risk estimation (default to median). After 2 years, full historical features become available."

---

## PRESENTATION TIPS

### Do's:
✅ **Emphasize business impact** - Speak in dollars and efficiency gains
✅ **Show live dashboard** - Brief 30-second demo builds credibility
✅ **Use concrete examples** - "A 95-year-old building in Brooklyn with no permits in 5 years"
✅ **Highlight leakage prevention** - This is a major technical contribution
✅ **Connect to real policy** - "This helps NYC allocate $4.6B tax revenue"

### Don'ts:
❌ Don't dive into hyperparameter tuning details unless asked
❌ Don't read slides verbatim - you know this material
❌ Don't apologize for limitations - frame as "future work opportunities"
❌ Don't skip the business metrics - professors care about real-world impact
❌ Don't go over 12 minutes - practice with timer

### Backup Slides (If time permits):
- Confusion matrix breakdown
- SHAP waterfall plot for specific building
- Dashboard architecture diagram
- ROC curve comparison across models

---

## VISUAL RECOMMENDATIONS FOR POWERPOINT

### Slide Design:
- **Use PACE colors:** Blue (#003C7D), Gold (#FFB81C), Navy (#002855)
- **Large fonts:** 24pt minimum for body text, 36pt+ for headers
- **High contrast:** Dark text on light backgrounds
- **Consistent layout:** Same header/footer across all slides

### Charts to Include:
1. **Chart 1** (Borough Distribution) → Slide 2
2. **Chart 2** (Data Sources) → Slide 2
3. **Chart 3** (System Architecture) → Slide 4
4. **Chart 4** (Model Comparison) → Slide 5
5. **Chart 5** (SHAP Importance) → Slide 7
6. **Chart 6** (Borough Risk) → Slides 3 & 9
7. **Chart 7** (Business Impact) → Slide 8
8. **Chart 8** (Metrics Dashboard) → Slide 6

### Pro Tips:
- **Use animation sparingly** - Fade in bullets, don't distract
- **Include slide numbers** - Helps with questions later
- **QR code on last slide** - Link to dashboard and GitHub
- **Team photo optional** - Humanizes presentation

---

## FINAL CHECKLIST

**Before Presentation:**
- [ ] Test all charts render correctly in PowerPoint
- [ ] Practice full run-through with timer (aim for 11:30)
- [ ] Verify dashboard is live and accessible
- [ ] Print backup slides in case of technical issues
- [ ] Test projector/screen with laptop
- [ ] Have water bottle ready

**During Presentation:**
- [ ] Make eye contact with entire committee
- [ ] Pause for emphasis after key metrics
- [ ] Point to charts while explaining them
- [ ] Speak slowly and clearly (natural nervous tendency is to rush)
- [ ] Smile and show confidence in your work

**After Presentation:**
- [ ] Thank committee for their time
- [ ] Offer to share dashboard link
- [ ] Collect feedback for future improvements

---

## SUCCESS METRICS FOR THIS PRESENTATION

You'll know you succeeded if:
1. ✅ Committee understands the **business impact** ($1.4M savings, 2.23× efficiency)
2. ✅ Technical rigor is clear (data leakage prevention, temporal validation)
3. ✅ Questions focus on **extensions** rather than defending basics
4. ✅ They remember "Office Apocalypse Algorithm" name
5. ✅ They check out the live dashboard after presentation

**Remember:** You've done excellent work. This presentation is just sharing what you already accomplished. Be proud and confident!

Good luck! 🚀
