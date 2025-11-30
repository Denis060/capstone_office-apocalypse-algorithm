# Strategic Questions for Professor Meeting
**Office Apocalypse Algorithm Project**

---

## 🎯 **Primary Questions (Must Ask)**

### **1. Performance Validation & Expectations**
**"Our baseline logistic regression achieves 88.2% ROC-AUC and 87.5% Precision@10%. For NYC policy applications, is this performance level acceptable, or should we target specific thresholds?"**

*Why this matters:* Sets performance expectations and guides whether we need advanced models

**Follow-up:** "What's more important for city planning - overall accuracy or precision in identifying high-risk buildings?"

### **2. Model Complexity Strategy**
**"Given our solid baseline, should we prioritize advancing to ensemble methods (Random Forest, XGBoost) for potentially higher performance, or focus on optimizing and interpreting our current model?"**

*Why this matters:* Determines our next phase strategy and time allocation

**Follow-up:** "How do you weigh the trade-off between model performance and explainability for policy applications?"

### **3. Data Quality & Temporal Validation**
**"We discovered and fixed critical data leakage that inflated performance from 99% to 88%. This required removing many engineered features. Are we being too conservative, or is this methodological rigor the right approach?"**

*Why this matters:* Validates our data cleaning approach and shows our analytical thinking

**Follow-up:** "Should we explore additional NYC datasets or focus on better feature engineering with existing data?"

### **4. Binary Classification with Three Risk Categories - CLARIFICATION NEEDED**
**"You mentioned confusion about our binary classification producing three risk levels. Let me clarify: we perform binary classification (High-risk vs Low-risk) but then categorize the probability scores into three business-friendly risk levels. Is this approach clear, or should we present it differently?"**

*Technical explanation:* 
- **Binary Classification**: Model predicts P(High Risk) vs P(Low Risk) 
- **Probability Scores**: Each building gets 0.0-1.0 probability of being high-risk
- **Risk Categories**: We bin probabilities into Low (<30%), Medium (30-70%), High (>70%) for business presentation

*Why this matters:* Ensures clear communication about our model architecture and outputs

**Follow-up:** "Should we stick with binary outputs, or would you prefer we present only the probability scores without risk categories?"

### **5. Timeline & Scope Prioritization**
**"Given our remaining tasks (hyperparameter tuning, advanced models, SHAP analysis, technical paper), what should be our priority order for the final weeks?"**

*Why this matters:* Ensures we deliver the most valuable components first

**Follow-up:** "Is it better to have one well-tuned model or comparison across multiple algorithms?"

---

## 🔍 **Secondary Questions (If Time Permits)**

### **5. Real-World Implementation**
"For actual NYC deployment, what performance metrics would city planners prioritize? Precision@10% for targeted interventions, or broader recall to catch all at-risk buildings?"

### **6. Business Application Context**
"Should our model aim to predict immediate vacancy risk, or longer-term vulnerability? This affects our feature selection and evaluation approach."

### **7. Technical Methodology**
"Our temporal validation uses synthetic dates due to data limitations. Should we invest time finding real temporal data, or is the current approach sufficient for demonstration?"

### **8. Academic vs Practical Focus**
"For the final deliverable, should we emphasize technical innovation or practical policy applicability?"

---

## 💡 **Questions That Show We're Thinking Ahead**

### **9. Scalability & Maintenance**
"If this model were actually deployed, how would we handle model drift and retraining with new building data?"

### **10. Ethical Considerations**
"Are there bias concerns we should address, particularly around building type, location, or ownership patterns?"

### **11. Validation Strategy**
"Would you recommend additional validation approaches beyond our temporal splits, such as geographic holdout or building type stratification?"

### **12. Feature Interpretation**
"For policy makers, should we focus on individual building risk scores or neighborhood-level insights?"

---

## 🎭 **Questions That Demonstrate Domain Knowledge**

### **13. NYC-Specific Context**
"Given NYC's unique real estate dynamics post-COVID, should we adjust our modeling approach to account for commercial real estate shifts?"

### **14. Policy Integration**
"How could this model integrate with existing NYC planning tools and databases for maximum impact?"

### **15. Comparative Analysis**
"Are there similar predictive models used in other cities that we should benchmark against?"

---

## 🚀 **Strategic Question Sequence**

### **Opening (Establish Status)**
1. Start with performance validation question (#1)
2. Show data quality discovery (#3) 
3. Ask for strategic direction (#2)

### **Middle (Deep Dive)**
4. Timeline and priorities (#4)
5. Real-world context (#5, #6)
6. Technical methodology validation (#7)

### **Closing (Forward-Looking)**
7. Academic focus (#8)
8. Implementation considerations (#9, #14)
9. Request specific feedback on next steps

---

## 📝 **How to Frame Each Question**

### **Structure:**
1. **Context:** "We found/achieved/discovered..."
2. **Question:** "Should we... or would you recommend..."
3. **Why it matters:** "This affects our..."

### **Examples:**

**Good:** "Our baseline model achieves 88% ROC-AUC with clean data after removing leaky features. Should we prioritize optimizing this model or exploring ensemble methods, given our timeline constraints?"

**Better:** "We discovered critical data leakage that inflated performance to 99%. After removing leaky features, we achieved 88% ROC-AUC - which demonstrates methodological rigor but required significant feature reduction. Should we continue this conservative approach or explore ways to safely incorporate more sophisticated features?"

---

## 🎯 **Expected Outcomes from Questions**

### **Performance Guidance**
- Clear performance thresholds
- Priority between accuracy and interpretability
- Validation approach confirmation

### **Strategic Direction**
- Next phase approach (optimization vs exploration)
- Time allocation guidance
- Feature engineering strategy

### **Academic Value**
- Research contribution clarity
- Technical innovation vs practical application
- Publication/presentation potential

---

## 🎪 **Backup Questions (If Discussion Stalls)**

1. "What's the most important lesson you'd want us to learn from this project?"
2. "Are there any red flags or common mistakes we should watch out for in the remaining work?"
3. "How does our approach compare to industry best practices for this type of problem?"
4. "What would make this project stand out in your experience?"
5. "Should we be preparing any specific materials for final presentation?"

---

**Remember:** The goal is to show we're actively thinking, working systematically, and need guidance on strategic decisions - not basic concepts!