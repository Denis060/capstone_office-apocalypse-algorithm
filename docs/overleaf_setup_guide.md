# Office Apocalypse Algorithm - Overleaf Integration Guide

## 📄 **Overleaf Setup Instructions**

### **1. Upload to Overleaf**
1. Create new project in Overleaf
2. Upload `overleaf_latex_template.tex` as your main document
3. Upload any figures to a `figures/` folder

### **2. Required LaTeX Packages**
The template includes all necessary packages:
```latex
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage[colorlinks=true, allcolors=blue]{hyperref}
\usepackage{booktabs}  % Professional tables
\usepackage{array}
\usepackage{multirow}
\usepackage{geometry}
\usepackage{fancyhdr}  # Headers/footers
\usepackage{setspace}  # Line spacing
\usepackage{caption}
\usepackage{subcaption}
```

### **3. Team Information Properly Formatted**

**Authors:** Ibrahim Denis Fofanah, Bright Arowny Zaman, and Jeevan Hemanth Yendluri

**Emails:** {if57774n, bz75499n, jy44272n}@pace.edu

**Institution:** Seidenberg School of Computer Science and Information Systems, Pace University

**Faculty Advisor:** Dr. Krishna Bathula

### **4. Key Tables Already Formatted**

#### **Table 1: Champion Model Performance**
- ROC-AUC: 92.41%
- Precision@10%: 93.01% 
- Precision@5%: 95.12%

#### **Table 2: SHAP Feature Importance**
- Building Age: 1.406 (Modernization incentives)
- Construction Activity: 1.149 (Economic development focus)
- Office Area: 0.776 (Space optimization)

#### **Table 3: Borough Risk Distribution**  
- Brooklyn: 40.9% high-risk (Highest)
- Manhattan: 22.1% high-risk (Lowest)

### **5. Figures to Add**

**Recommended Figures for Upload:**
```
figures/
├── system_architecture.png     # Your system diagram
├── shap_summary_plot.png       # SHAP feature importance
├── shap_importance_bar.png     # SHAP bar chart  
├── geographic_risk_map.png     # NYC borough risk map
├── dashboard_screenshot.png    # Streamlit dashboard
└── model_comparison.png        # XGBoost vs RF vs LR
```

### **6. Bibliography Setup**

Create `references.bib` file:
```bibtex
@inproceedings{chen2016xgboost,
  title={Xgboost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD},
  pages={785--794},
  year={2016}
}

@article{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@misc{nyc_pluto_2025,
  title={Property Assessment Data (PLUTO)},
  author={{NYC Department of Finance}},
  howpublished={NYC Open Data Portal},
  year={2025},
  url={https://opendata.cityofnewyork.us/}
}
```

### **7. Content Sections to Expand**

**From your technical_paper_draft2.md, add:**

1. **Section 2: Related Work** - Literature review section
2. **Section 3.1-3.2: System Architecture** - Complete technical details  
3. **Section 4: Full Analysis & Results** - All your comprehensive findings
4. **Section 5: Complete Conclusions** - All 6 key contributions

### **8. Professional Formatting Features**

**Already Included:**
- ✅ Professional title formatting
- ✅ Team author information
- ✅ Abstract with keywords
- ✅ Table formatting with booktabs
- ✅ Figure referencing system
- ✅ IEEE bibliography style
- ✅ Page headers and numbering
- ✅ 1.5 line spacing
- ✅ Proper margins (1 inch)

## 🚀 **Next Steps for Overleaf**

1. **Upload** the LaTeX template to your Overleaf project
2. **Copy content** from `technical_paper_draft2.md` into appropriate LaTeX sections  
3. **Add figures** from your analysis and dashboard
4. **Create bibliography** with your references
5. **Compile** and review the formatted paper

The template is ready for immediate use in Overleaf with your complete team information and professional academic formatting! 📚