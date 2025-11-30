"""
Clean Model Testing - No Data Leakage

This script tests our models with the clean dataset that has all
leakage features removed to get realistic performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def test_clean_models():
    """Test models with clean dataset (no data leakage)."""
    print("CLEAN MODEL TESTING - NO DATA LEAKAGE")
    print("=" * 45)
    
    # Load clean dataset
    data = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Loaded clean data: {len(data):,} records")
    print(f"Features: {data.shape[1] - 2}")  # -2 for target and BBL
    
    # Prepare features
    feature_cols = [col for col in data.columns 
                   if col not in ['target_high_vacancy_risk', 'BBL']]
    
    X = data[feature_cols].copy()
    y = data['target_high_vacancy_risk'].copy()
    
    print(f"\\nTarget distribution:")
    print(f"Class 0 (low risk): {(y == 0).sum():,} ({(y == 0).mean():.3f})")
    print(f"Class 1 (high risk): {(y == 1).sum():,} ({(y == 1).mean():.3f})")
    
    # Handle missing values
    print(f"\\nMissing values before cleaning: {X.isnull().sum().sum()}")
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            if X[col].isnull().sum() > 0:
                # Use median, but fallback to 0 if median is NaN
                median_val = X[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0
                X[col] = X[col].fillna(fill_val)
                print(f"  Filled {col} with {fill_val}")
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Final cleanup - any remaining NaN gets filled with 0
    remaining_nas = X.isnull().sum().sum()
    if remaining_nas > 0:
        print(f"  Filling remaining {remaining_nas} NaN values with 0")
        X = X.fillna(0)
    
    print(f"Missing values after cleaning: {X.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Feature set: {X.shape[1]} features")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\\nTrain set: {len(X_train):,} records")
    print(f"Test set: {len(X_test):,} records")
    
    # Initialize models
    models = {
        'logistic': {
            'name': 'Logistic Regression',
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'scaler': StandardScaler()
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=100, random_state=42, 
                class_weight='balanced'
            ),
            'scaler': None
        }
    }
    
    # Train and evaluate models
    results = {}
    
    for model_key, config in models.items():
        print(f"\\n{'-' * 40}")
        print(f"TRAINING {config['name'].upper()}")
        print(f"{'-' * 40}")
        
        model = config['model']
        scaler = config['scaler']
        
        # Prepare training data
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted'),
            'roc_auc': roc_auc_score(y_train, y_pred_proba_train)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_test)
        }
        
        # Precision@K metrics
        for k in [0.05, 0.1, 0.2, 0.3]:
            threshold = np.percentile(y_pred_proba_test, 100 * (1 - k))
            top_k_pred = (y_pred_proba_test >= threshold).astype(int)
            if np.sum(top_k_pred) > 0:
                precision_k = precision_score(y_test, top_k_pred, zero_division=0)
            else:
                precision_k = 0.0
            test_metrics[f'precision_at_{k}'] = precision_k
        
        results[model_key] = {
            'name': config['name'],
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_proba': y_pred_proba_test
        }
        
        # Print results
        print(f"\\nTraining Performance:")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
        
        print(f"\\nTest Performance:")
        print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"  Precision:    {test_metrics['precision']:.4f}")
        print(f"  Recall:       {test_metrics['recall']:.4f}")
        print(f"  F1-Score:     {test_metrics['f1']:.4f}")
        print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}")
        print(f"  Precision@5%: {test_metrics['precision_at_0.05']:.4f}")
        print(f"  Precision@10%:{test_metrics['precision_at_0.1']:.4f}")
        
        # Feature importance for Random Forest
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\\nTop 10 Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create comparison plots
    create_comparison_plots(results, y_test, X_test.columns)
    
    # Summary comparison
    print(f"\\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY (CLEAN DATA - NO LEAKAGE)")
    print(f"{'=' * 60}")
    
    comparison_data = []
    for model_key, result in results.items():
        metrics = result['test_metrics']
        comparison_data.append({
            'Model': result['name'],
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'Precision@10%': metrics['precision_at_0.1'],
            'F1-Score': metrics['f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\\nModel Performance Comparison:")
    print("-" * 50)
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<20} | ACC: {row['Accuracy']:.3f} | "
              f"AUC: {row['ROC-AUC']:.3f} | P@10%: {row['Precision@10%']:.3f}")
    
    # Save results
    comparison_df.to_csv('results/clean_model_comparison.csv', index=False)
    
    print(f"\\nREALISTIC PERFORMANCE WITHOUT DATA LEAKAGE:")
    print("=" * 50)
    best_model = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
    print(f"Champion Model: {best_model['Model']}")
    print(f"  Accuracy:      {best_model['Accuracy']:.3f} (vs 99.6% with leakage)")
    print(f"  ROC-AUC:       {best_model['ROC-AUC']:.3f} (vs 99.9% with leakage)")
    print(f"  Precision@10%: {best_model['Precision@10%']:.3f} (vs 100% with leakage)")
    
    return results, comparison_df

def create_comparison_plots(results, y_test, feature_names):
    """Create comparison plots for clean models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clean Models Performance (No Data Leakage)', fontsize=16, fontweight='bold')
    
    colors = {'logistic': 'blue', 'random_forest': 'green'}
    
    # ROC Curves
    ax = axes[0, 0]
    for model_key, result in results.items():
        y_pred_proba = result['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = result['test_metrics']['roc_auc']
        
        ax.plot(fpr, tpr, color=colors[model_key], linewidth=2,
               label=f"{result['name']} (AUC = {auc_score:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Clean Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    ax = axes[0, 1]
    for model_key, result in results.items():
        y_pred_proba = result['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        ax.plot(recall, precision, color=colors[model_key], linewidth=2,
               label=f"{result['name']}")
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Model Performance Comparison
    ax = axes[1, 0]
    metrics = ['accuracy', 'roc_auc', 'precision_at_0.1']
    metric_labels = ['Accuracy', 'ROC-AUC', 'Precision@10%']
    
    model_names = []
    metric_values = {metric: [] for metric in metrics}
    
    for model_key, result in results.items():
        model_names.append(result['name'])
        for metric in metrics:
            metric_values[metric].append(result['test_metrics'][metric])
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax.bar(x_pos + i * width, metric_values[metric], width, 
              label=metric_labels[i], alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training vs Test Performance
    ax = axes[1, 1]
    train_aucs = []
    test_aucs = []
    model_names = []
    
    for model_key, result in results.items():
        model_names.append(result['name'])
        train_aucs.append(result['train_metrics']['roc_auc'])
        test_aucs.append(result['test_metrics']['roc_auc'])
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x_pos - width/2, train_aucs, width, label='Training AUC', alpha=0.7, color='lightblue')
    ax.bar(x_pos + width/2, test_aucs, width, label='Test AUC', alpha=0.7, color='lightcoral')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Train vs Test Performance (Overfitting Check)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/clean_models_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\\nSaved evaluation plots: results/clean_models_evaluation.png")
    plt.close()

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Test clean models
    results, comparison = test_clean_models()