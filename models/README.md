# Model Artifacts Directory
**Office Apocalypse Algorithm - Trained Models and Data**

## 📋 Overview
This directory contains all trained machine learning models and the exact training/test data splits used for the Office Apocalypse Algorithm. These artifacts allow for complete model reproducibility and deployment.

## 🗂️ Files Description

### Training & Test Data
- **`X_train.csv`** (8.7 MB) - Training features
  - **Records**: 5,752 office buildings (80% of total)
  - **Features**: 76 selected features after variance filtering
  - **Format**: Scaled features ready for ML models
  - **Use**: Training data that was used to train all models

- **`X_test.csv`** (2.2 MB) - Test features
  - **Records**: 1,439 office buildings (20% of total)
  - **Features**: 76 selected features (same as training)
  - **Format**: Scaled features using training scaler
  - **Use**: Test data for unbiased model evaluation

- **`y_train.csv`** (45 KB) - Training labels
  - **Records**: 5,752 labels
  - **Target**: Binary vacancy risk (0=Low Risk, 1=High Risk)
  - **Distribution**: 20.0% positive class (high risk)

- **`y_test.csv`** (11 KB) - Test labels
  - **Records**: 1,439 labels
  - **Target**: Binary vacancy risk (0=Low Risk, 1=High Risk)
  - **Distribution**: 20.0% positive class (preserved from training)

### Feature Engineering
- **`feature_scaler.joblib`** (3.8 KB) - Fitted StandardScaler
  - **Purpose**: Transforms raw features to standardized scale
  - **Fitted on**: Training data only (prevents data leakage)
  - **Usage**: Apply to new data before prediction

### Trained Models
- **`champion_model.joblib`** (2.8 KB) - Best performing model
  - **Model**: Logistic Regression (best test performance)
  - **Performance**: 99.99% ROC-AUC, 98.75% Accuracy
  - **Usage**: Primary model for predictions

- **`logistic_regression_model.joblib`** (2.8 KB) - Logistic Regression
  - **Performance**: 99.99% ROC-AUC, 98.75% Accuracy, 100% Recall
  - **Strengths**: Perfect recall, excellent interpretability

- **`gradient_boosting_model.joblib`** (592 KB) - Gradient Boosting
  - **Performance**: 99.91% ROC-AUC, 98.54% Accuracy
  - **Strengths**: Strong ensemble performance

- **`hist_gradient_boosting_model.joblib`** (462 KB) - Hist Gradient Boosting
  - **Performance**: 99.91% ROC-AUC, 98.75% Accuracy
  - **Strengths**: Fast training, good performance

- **`random_forest_model.joblib`** (1.6 MB) - Random Forest
  - **Performance**: 99.50% ROC-AUC, 95.48% Accuracy
  - **Strengths**: Robust, good feature importance

### Metadata
- **`model_metadata.json`** (2.3 KB) - Complete training information
  - Training date and parameters
  - Champion model details and performance
  - Feature list and data sources
  - Class distribution and sample counts

## 📊 Key Statistics
- **Total Buildings**: 7,191 NYC office buildings
- **Training Split**: 5,752 buildings (80%)
- **Test Split**: 1,439 buildings (20%)
- **Features Used**: 76 engineered features
- **Champion Model**: Logistic Regression
- **Best Performance**: 99.99% ROC-AUC, 98.75% Accuracy

## 🚀 Usage Examples

### Load Champion Model
```python
import joblib
import pandas as pd

# Load the champion model and scaler
champion_model = joblib.load('models/champion_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

# Load test data to verify
X_test = pd.read_csv('models/X_test.csv', index_col=0)
y_test = pd.read_csv('models/y_test.csv', index_col=0)

# Make predictions
predictions = champion_model.predict(X_test)
probabilities = champion_model.predict_proba(X_test)[:, 1]
```

### Load Training Data
```python
# Load the exact training data used
X_train = pd.read_csv('models/X_train.csv', index_col=0)
y_train = pd.read_csv('models/y_train.csv', index_col=0)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"High risk rate: {y_train.mean():.1%}")
```

### Load Model Metadata
```python
import json

# Load complete model information
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Champion model: {metadata['champion_model']}")
print(f"Training date: {metadata['training_date']}")
print(f"ROC-AUC: {metadata['champion_performance']['roc_auc']:.4f}")
```

## 🔄 Reproducibility
All models can be perfectly reproduced using:
1. Raw data from `data/raw/`
2. Feature engineering from `data/features/`
3. Training pipeline from `notebooks/03_model_training.ipynb`
4. Exact train/test splits preserved in this directory

## 📈 Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **98.75%** | **94.12%** | **100.00%** | **96.97%** | **99.99%** |
| Hist Gradient Boosting | 98.75% | 96.23% | 97.57% | 96.90% | 99.91% |
| Gradient Boosting | 98.54% | 96.52% | 96.18% | 96.35% | 99.91% |
| Random Forest | 95.48% | 82.13% | 98.96% | 89.76% | 99.50% |

## 📝 Notes
- All models were trained with stratified sampling to preserve class balance
- Feature scaling was applied consistently across train/test splits
- No data leakage: scaler fitted only on training data
- Geographic stratification ensured representative sampling

---
*Generated on: October 6, 2025*
*Training completed: Office Apocalypse Algorithm v1.0*