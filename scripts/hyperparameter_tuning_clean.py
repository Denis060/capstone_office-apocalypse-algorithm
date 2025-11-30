#!/usr/bin/env python3
"""
Hyperparameter Tuning for Clean Models
Office Apocalypse Algorithm - No Data Leakage

This script optimizes Random Forest and Logistic Regression models
using only clean features without data leakage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def precision_at_k(y_true, y_prob, k):
    """Calculate precision@k for targeting top k% of predictions."""
    n = len(y_true)
    k_count = int(n * k / 100)
    top_k_indices = np.argsort(y_prob)[-k_count:]
    return np.mean(y_true.iloc[top_k_indices])

def load_clean_data():
    """Load the clean dataset without data leakage."""
    print("Loading clean dataset...")
    df = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Loaded clean data: {len(df)} records")
    
    # Create target variable  
    if 'target_high_vacancy_risk' not in df.columns:
        if 'vacancy_risk_alert' in df.columns:
            df['target_high_vacancy_risk'] = (
                df['vacancy_risk_alert'].isin(['Orange', 'Red'])
            ).astype(int)
        else:
            raise ValueError("Cannot create target variable")
    
    # Clean features (no composite/derived features)
    safe_features = [
        'building_age', 'lotarea', 'bldgarea', 'officearea', 'numfloors',
        'assessland', 'assesstot', 'yearbuilt', 'value_per_sqft',
        'office_ratio', 'floor_efficiency', 'land_value_ratio',
        'transaction_count', 'deed_count', 'mortgage_count',
        'mta_accessibility_proxy', 'business_density_proxy',
        'construction_activity_proxy', 'commercial_ratio', 'distress_score'
    ]
    
    # Filter to available features
    available_features = [f for f in safe_features if f in df.columns]
    print(f"Using {len(available_features)} clean features")
    
    # Handle missing values
    print("Handling missing values...")
    df_clean = df.copy()
    
    for feature in available_features:
        if df_clean[feature].isna().sum() > 0:
            if df_clean[feature].dtype in ['float64', 'int64']:
                # Use median for numeric features
                fill_value = df_clean[feature].median()
                if pd.isna(fill_value):  # If median is also NaN, use 0
                    fill_value = 0
                df_clean[feature].fillna(fill_value, inplace=True)
                print(f"  Filled {feature} with {fill_value}")
            else:
                # Use mode for categorical features
                mode_values = df_clean[feature].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                df_clean[feature].fillna(fill_value, inplace=True)
                print(f"  Filled {feature} with {fill_value}")
    
    # Replace any remaining infinite values
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill any remaining NaNs with 0
    for feature in available_features:
        if df_clean[feature].isna().sum() > 0:
            df_clean[feature].fillna(0, inplace=True)
            print(f"  Final fill for {feature} with 0")
    
    # Final check
    print(f"Missing values after cleaning: {df_clean[available_features].isna().sum().sum()}")
    print(f"Infinite values: {np.isinf(df_clean[available_features]).sum().sum()}")
    
    X = df_clean[available_features]
    y = df_clean['target_high_vacancy_risk']
    
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    
    return X, y, available_features

def tune_logistic_regression(X_train, X_test, y_train, y_test):
    """Tune Logistic Regression hyperparameters."""
    print("\n" + "="*50)
    print("TUNING LOGISTIC REGRESSION")
    print("="*50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Parameter grid - simplified to avoid solver conflicts
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  # Works with both l1 and l2
        'max_iter': [1000, 2000]
    }
    
    # Grid search with cross-validation
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    
    print("Running grid search...")
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_lr = grid_search.best_estimator_
    
    # Add calibration
    calibrated_lr = CalibratedClassifierCV(best_lr, method='isotonic', cv=3)
    calibrated_lr.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = calibrated_lr.predict(X_test_scaled)
    y_prob = calibrated_lr.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results = {
        'model': 'Logistic Regression (Tuned)',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision_at_5': precision_at_k(y_test, y_prob, 5),
        'precision_at_10': precision_at_k(y_test, y_prob, 10),
        'precision_at_20': precision_at_k(y_test, y_prob, 20)
    }
    
    print(f"\nTuned Logistic Regression Results:")
    print(f"  Accuracy:      {results['accuracy']:.4f}")
    print(f"  ROC-AUC:       {results['roc_auc']:.4f}")
    print(f"  Precision@10%: {results['precision_at_10']:.4f}")
    
    return calibrated_lr, scaler, results

def tune_random_forest(X_train, X_test, y_train, y_test):
    """Tune Random Forest hyperparameters."""
    print("\n" + "="*50)
    print("TUNING RANDOM FOREST")
    print("="*50)
    
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Initial coarse grid search
    print("Running coarse grid search...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Reduced grid for speed
    coarse_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', None],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        rf, coarse_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best coarse parameters: {grid_search.best_params_}")
    print(f"Best coarse CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Fine-tune around best parameters
    best_params = grid_search.best_params_
    
    # Build final model
    best_rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    
    best_rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    results = {
        'model': 'Random Forest (Tuned)',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision_at_5': precision_at_k(y_test, y_prob, 5),
        'precision_at_10': precision_at_k(y_test, y_prob, 10),
        'precision_at_20': precision_at_k(y_test, y_prob, 20)
    }
    
    print(f"\nTuned Random Forest Results:")
    print(f"  Accuracy:      {results['accuracy']:.4f}")
    print(f"  ROC-AUC:       {results['roc_auc']:.4f}")
    print(f"  Precision@10%: {results['precision_at_10']:.4f}")
    
    return best_rf, results

def create_comparison_plots(lr_results, rf_results):
    """Create comparison plots for both models."""
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model comparison
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [lr_results['accuracy'], rf_results['accuracy']]
    roc_aucs = [lr_results['roc_auc'], rf_results['roc_auc']]
    precision_10 = [lr_results['precision_at_10'], rf_results['precision_at_10']]
    
    # Accuracy comparison
    axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightgreen'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0.7, 0.9)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # ROC-AUC comparison
    axes[0, 1].bar(models, roc_aucs, color=['skyblue', 'lightgreen'])
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_ylim(0.8, 1.0)
    for i, v in enumerate(roc_aucs):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Precision@10% comparison
    axes[1, 0].bar(models, precision_10, color=['skyblue', 'lightgreen'])
    axes[1, 0].set_title('Precision@10% Comparison')
    axes[1, 0].set_ylabel('Precision@10%')
    axes[1, 0].set_ylim(0.8, 1.0)
    for i, v in enumerate(precision_10):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Performance summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = [
        ['Model', 'Accuracy', 'ROC-AUC', 'Precision@10%'],
        ['Logistic Regression', f"{lr_results['accuracy']:.3f}", f"{lr_results['roc_auc']:.3f}", f"{lr_results['precision_at_10']:.3f}"],
        ['Random Forest', f"{rf_results['accuracy']:.3f}", f"{rf_results['roc_auc']:.3f}", f"{rf_results['precision_at_10']:.3f}"]
    ]
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved to: results/hyperparameter_tuning_results.png")

def save_champion_model(best_model, scaler, model_name, feature_names):
    """Save the champion model for deployment."""
    print(f"\nSaving champion model: {model_name}")
    
    # Save model
    joblib.dump(best_model, f'models/champion_{model_name.lower().replace(" ", "_")}.pkl')
    
    if scaler:
        joblib.dump(scaler, f'models/champion_scaler.pkl')
    
    # Save feature names
    with open(f'models/champion_features.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Model saved to: models/champion_{model_name.lower().replace(' ', '_')}.pkl")

def main():
    """Main hyperparameter tuning pipeline."""
    print("HYPERPARAMETER TUNING FOR CLEAN MODELS")
    print("=" * 60)
    
    # Load clean data
    X, y, feature_names = load_clean_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    # Tune models
    tuned_lr, scaler, lr_results = tune_logistic_regression(X_train, X_test, y_train, y_test)
    tuned_rf, rf_results = tune_random_forest(X_train, X_test, y_train, y_test)
    
    # Comparison
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"Logistic Regression: ROC-AUC {lr_results['roc_auc']:.4f}, Accuracy {lr_results['accuracy']:.4f}")
    print(f"Random Forest:       ROC-AUC {rf_results['roc_auc']:.4f}, Accuracy {rf_results['accuracy']:.4f}")
    
    # Determine champion
    if rf_results['roc_auc'] > lr_results['roc_auc']:
        champion_model = tuned_rf
        champion_name = "Random Forest"
        champion_scaler = None
        print(f"\n🏆 CHAMPION MODEL: Random Forest (ROC-AUC: {rf_results['roc_auc']:.4f})")
    else:
        champion_model = tuned_lr
        champion_name = "Logistic Regression"
        champion_scaler = scaler
        print(f"\n🏆 CHAMPION MODEL: Logistic Regression (ROC-AUC: {lr_results['roc_auc']:.4f})")
    
    # Save results
    create_comparison_plots(lr_results, rf_results)
    save_champion_model(champion_model, champion_scaler, champion_name, feature_names)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("  - results/hyperparameter_tuning_results.png")
    print("  - models/champion_*.pkl")
    
    return champion_model, champion_name

if __name__ == "__main__":
    main()