"""
Quick Hyperparameter Tuning for Random Forest

This script runs a focused hyperparameter search to optimize 
the Random Forest model quickly and efficiently.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, make_scorer
import time
import joblib

def precision_at_k_scorer(k=0.1):
    """Custom scorer for Precision@K."""
    def precision_at_k(y_true, y_proba):
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        threshold = np.percentile(y_proba, 100 * (1 - k))
        top_k_pred = (y_proba >= threshold).astype(int)
        if np.sum(top_k_pred) == 0:
            return 0.0
        return precision_score(y_true, top_k_pred, zero_division=0)
    return make_scorer(precision_at_k, needs_proba=True)

def quick_hyperparameter_tuning():
    """Run focused hyperparameter tuning."""
    print("QUICK HYPERPARAMETER TUNING - RANDOM FOREST")
    print("=" * 45)
    
    # Load clean data
    data = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Loaded data: {len(data):,} records")
    
    # Prepare features
    feature_cols = [col for col in data.columns 
                   if col not in ['target_high_vacancy_risk', 'BBL']]
    X = data[feature_cols].copy()
    y = data['target_high_vacancy_risk'].copy()
    
    # Clean data
    print("\\nCleaning data...")
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64'] and X[col].isnull().sum() > 0:
            median_val = X[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            X[col] = X[col].fillna(fill_val)
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Clean dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Split for tuning
    X_tune, X_holdout, y_tune, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tuning set: {len(X_tune):,} samples")
    print(f"Holdout set: {len(X_holdout):,} samples")
    
    # Focused parameter grid (smaller for speed)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 15, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', None]
    }
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\\nParameter combinations to test: {total_combinations}")
    print("Estimated time: ~10-15 minutes")
    
    # Setup models and CV
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3
    
    # Multiple scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'precision_at_10': precision_at_k_scorer(0.1)
    }
    
    # Run grid search
    print("\\nRunning grid search...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_tune, y_tune)
    
    elapsed_time = time.time() - start_time
    print(f"\\nGrid search completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Best results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\\nBEST PARAMETERS:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\\nBEST CROSS-VALIDATION SCORE:")
    print(f"  ROC-AUC: {best_score:.4f}")
    
    # Test on holdout
    print("\\nHOLDOUT SET EVALUATION:")
    y_pred_holdout = best_model.predict(X_holdout)
    y_pred_proba_holdout = best_model.predict_proba(X_holdout)[:, 1]
    
    holdout_metrics = {
        'accuracy': accuracy_score(y_holdout, y_pred_holdout),
        'roc_auc': roc_auc_score(y_holdout, y_pred_proba_holdout),
        'precision': precision_score(y_holdout, y_pred_holdout, average='weighted'),
    }
    
    # Precision@K
    for k in [0.05, 0.1, 0.2]:
        threshold = np.percentile(y_pred_proba_holdout, 100 * (1 - k))
        top_k_pred = (y_pred_proba_holdout >= threshold).astype(int)
        if np.sum(top_k_pred) > 0:
            precision_k = precision_score(y_holdout, top_k_pred, zero_division=0)
        else:
            precision_k = 0.0
        holdout_metrics[f'precision_at_{k}'] = precision_k
    
    for metric, score in holdout_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nTOP 10 FEATURES (Tuned Model):")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create simple comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
    
    # Feature importance plot
    top_features = feature_importance.head(15)
    axes[0].barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.7)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Feature Importance (Tuned Model)')
    axes[0].grid(True, alpha=0.3)
    
    # Performance comparison
    metrics_names = ['ROC-AUC', 'Accuracy', 'Precision@10%']
    cv_scores = [best_score, 
                grid_search.cv_results_[f'mean_test_accuracy'][grid_search.best_index_],
                grid_search.cv_results_[f'mean_test_precision_at_10'][grid_search.best_index_]]
    holdout_scores = [holdout_metrics['roc_auc'], 
                     holdout_metrics['accuracy'], 
                     holdout_metrics['precision_at_0.1']]
    
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, cv_scores, width, label='CV Score', alpha=0.7, color='lightblue')
    axes[1].bar(x_pos + width/2, holdout_scores, width, label='Holdout Score', alpha=0.7, color='lightcoral')
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Score')
    axes[1].set_title('CV vs Holdout Performance')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quick_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print(f"\\nSaved plot: results/quick_hyperparameter_tuning.png")
    plt.close()
    
    # Save results
    print("\\nSaving results...")
    
    # Best model
    joblib.dump(best_model, 'results/tuned_random_forest_model.pkl')
    print("  ✅ Saved tuned model: results/tuned_random_forest_model.pkl")
    
    # Best parameters
    params_df = pd.DataFrame([best_params])
    params_df.to_csv('results/tuned_hyperparameters.csv', index=False)
    print("  ✅ Saved parameters: results/tuned_hyperparameters.csv")
    
    # Holdout results
    holdout_df = pd.DataFrame([holdout_metrics])
    holdout_df.to_csv('results/tuned_model_holdout_results.csv', index=False)
    print("  ✅ Saved holdout results: results/tuned_model_holdout_results.csv")
    
    # Feature importance
    feature_importance.to_csv('results/tuned_model_feature_importance.csv', index=False)
    print("  ✅ Saved feature importance: results/tuned_model_feature_importance.csv")
    
    # Summary comparison
    print("\\n" + "="*50)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*50)
    print(f"✅ Tuning completed in {elapsed_time/60:.1f} minutes")
    print(f"✅ Best CV ROC-AUC: {best_score:.4f}")
    print(f"✅ Holdout ROC-AUC: {holdout_metrics['roc_auc']:.4f}")
    print(f"✅ Holdout Precision@10%: {holdout_metrics['precision_at_0.1']:.4f}")
    print("\\n🎯 CHAMPION HYPERPARAMETERS:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    print("\\n🚀 Ready for Task 4.6 - Final Model Evaluation!")
    
    return best_model, best_params, holdout_metrics, feature_importance

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    model, params, metrics, importance = quick_hyperparameter_tuning()