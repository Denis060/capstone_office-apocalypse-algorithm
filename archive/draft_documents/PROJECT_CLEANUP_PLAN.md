# Project Structure Cleanup Plan

## Current Issues
1. **Multiple duplicate README files** - README.md, README_midterm.md, README_technical_paper.md
2. **Scattered documentation** - Multiple loose .md files in root directory
3. **Redundant source files** - advanced_models.py vs advanced_models_simplified.py
4. **Outdated files** - Midterm references, speaker notes
5. **Mixed file locations** - Some docs in root, some in docs/

## Proposed Clean Structure

```
office_apocalypse_algorithm_project/
├── README.md                          # Main project README
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
├── data/                             # Data files (keep as is)
├── src/                              # Core source code
│   ├── data_loader.py                # Data loading utilities
│   ├── temporal_validation.py        # Temporal validation framework
│   ├── baseline_model.py             # Baseline logistic regression
│   ├── advanced_models.py            # Advanced models (cleaned)
│   ├── hyperparameter_tuning.py      # Hyperparameter optimization
│   └── __pycache__/                  # Python cache (auto-generated)
├── scripts/                          # Analysis and utility scripts
│   ├── analyze_data_leakage.py       # Data leakage analysis
│   ├── test_clean_models.py          # Model testing
│   ├── tune_and_shap.py              # Tuning and interpretation
│   └── create_presentation_visuals.py # Visualization generation
├── notebooks/                        # Jupyter notebooks (keep current)
├── docs/                            # All documentation
│   ├── professor_presentation.md     # Meeting presentation
│   ├── meeting_talking_points.md     # Talking points
│   ├── professor_questions.md        # Q&A preparation
│   ├── PROJECT_OVERVIEW.md          # Consolidated project overview
│   ├── TECHNICAL_METHODOLOGY.md     # Technical details
│   └── visuals/                     # Documentation images
├── results/                         # Model outputs and analysis
├── figures/                         # Generated plots and charts
├── models/                          # Saved model files
└── archive/                         # Moved outdated files
    ├── midterm_materials/
    ├── draft_documents/
    └── deprecated_scripts/
```

## Files to Archive/Remove

### Move to archive/midterm_materials/
- midterm_references.md
- midterm_slides.md  
- README_midterm.md
- speaker_notes.md

### Move to archive/draft_documents/  
- EXECUTIVE_SUMMARY.md
- FINAL_REVIEW_CHECKLIST.md
- LESSONS_LEARNED.md
- ACCURACY_INVESTIGATION_PROFESSOR_MEETING.md
- DATASET_JUSTIFICATION.md
- PROFESSOR_DATASET_JUSTIFICATION.md
- PROJECT_POSTER.md
- technical_paper_draft1.md
- technical_paper_draft1.tex
- SYSTEM_STATUS.md
- README_technical_paper.md

### Move to archive/deprecated_scripts/
- src/advanced_models_simplified.py (keep advanced_models.py)
- src/modeling.py (functionality moved to other files)
- src/feature_engineering.py (if not actively used)
- src/quick_hyperparameter_tuning.py (keep main hyperparameter_tuning.py)
- scripts/analyze_contributions.py
- scripts/clean_model_analysis.py
- scripts/extract_notebook_images.py
- scripts/generate_sample_figures.py
- scripts/model_impact_analysis.py
- scripts/model_impact_clean.py
- scripts/model_robustness_analysis.py
- scripts/project_state_review.py

### Clean docs/ directory
- Move misc documentation to proper categories
- Consolidate related docs
- Create clear visual organization

## Action Items
1. Create archive/ directory structure
2. Move outdated files to archive
3. Consolidate documentation in docs/
4. Update main README.md with current project state
5. Remove duplicate/redundant source files
6. Clean up root directory