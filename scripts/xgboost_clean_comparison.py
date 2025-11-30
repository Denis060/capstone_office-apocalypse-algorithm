#!/usr/bin/env python3
"""
Clean XGBoost Comparison
Office Apocalypse Algorithm - No Data Leakage

Compare XGBoost performance against our champion Random Forest
using only clean features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import shap
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
                fill_value = df_clean[feature].median()
                if pd.isna(fill_value):
                    fill_value = 0
                df_clean[feature].fillna(fill_value, inplace=True)
            else:
                mode_values = df_clean[feature].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                df_clean[feature].fillna(fill_value, inplace=True)
    
    # Replace infinite values and final NaN cleanup
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    for feature in available_features:
        if df_clean[feature].isna().sum() > 0:
            df_clean[feature].fillna(0, inplace=True)
    
    print(f"Missing values after cleaning: {df_clean[available_features].isna().sum().sum()}")
    
    X = df_clean[available_features]
    y = df_clean['target_high_vacancy_risk']
    
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    
    return X, y, available_features

def train_xgboost_clean(X_train, X_test, y_train, y_test):
    """Train XGBoost with clean features and compare to Random Forest."""
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST WITH CLEAN FEATURES")
    print("="*60)
    
    # XGBoost with optimized parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle class imbalance
    )
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # XGBoost Results
    xgb_results = {
        'model': 'XGBoost (Clean)',
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_prob_xgb),
        'precision_at_5': precision_at_k(y_test, y_prob_xgb, 5),
        'precision_at_10': precision_at_k(y_test, y_prob_xgb, 10),
        'precision_at_20': precision_at_k(y_test, y_prob_xgb, 20)
    }
    
    print(f"\nXGBoost Results (Clean Features):")
    print(f"  Accuracy:      {xgb_results['accuracy']:.4f}")
    print(f"  ROC-AUC:       {xgb_results['roc_auc']:.4f}")
    print(f"  Precision@10%: {xgb_results['precision_at_10']:.4f}")
    
    return xgb_model, xgb_results

def train_random_forest_comparison(X_train, X_test, y_train, y_test):
    """Train Random Forest for comparison."""
    
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST FOR COMPARISON")
    print("="*60)
    
    # Use optimized parameters from hyperparameter tuning
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=4,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Random Forest Results
    rf_results = {
        'model': 'Random Forest (Clean)',
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_prob_rf),
        'precision_at_5': precision_at_k(y_test, y_prob_rf, 5),
        'precision_at_10': precision_at_k(y_test, y_prob_rf, 10),
        'precision_at_20': precision_at_k(y_test, y_prob_rf, 20)
    }
    
    print(f"\nRandom Forest Results (Clean Features):")
    print(f"  Accuracy:      {rf_results['accuracy']:.4f}")
    print(f"  ROC-AUC:       {rf_results['roc_auc']:.4f}")
    print(f"  Precision@10%: {rf_results['precision_at_10']:.4f}")
    
    return rf_model, rf_results

def create_comparison_plots(xgb_results, rf_results, X_test, y_test, xgb_model, rf_model):
    """Create comprehensive comparison plots."""
    
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Model comparison metrics
    models = ['Random Forest', 'XGBoost']
    accuracies = [rf_results['accuracy'], xgb_results['accuracy']]
    roc_aucs = [rf_results['roc_auc'], xgb_results['roc_auc']]
    precision_10 = [rf_results['precision_at_10'], xgb_results['precision_at_10']]
    
    # Accuracy comparison
    bars1 = axes[0, 0].bar(models, accuracies, color=['lightgreen', 'lightcoral'])
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0.7, 0.9)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # ROC-AUC comparison
    bars2 = axes[0, 1].bar(models, roc_aucs, color=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_ylim(0.8, 1.0)
    for i, v in enumerate(roc_aucs):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Precision@10% comparison
    bars3 = axes[0, 2].bar(models, precision_10, color=['lightgreen', 'lightcoral'])
    axes[0, 2].set_title('Precision@10% Comparison', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Precision@10%')
    axes[0, 2].set_ylim(0.8, 1.0)
    for i, v in enumerate(precision_10):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # ROC Curves
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    
    axes[1, 0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_results["roc_auc"]:.3f})', color='green', linewidth=2)
    axes[1, 0].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_results["roc_auc"]:.3f})', color='red', linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Feature Importance Comparison
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    
    feature_names = X_test.columns
    top_n = 10
    
    # Get top features from XGBoost
    top_features_idx = np.argsort(xgb_importance)[-top_n:]
    top_features = feature_names[top_features_idx]
    
    rf_top_importance = rf_importance[top_features_idx]
    xgb_top_importance = xgb_importance[top_features_idx]
    
    x = np.arange(len(top_features))
    width = 0.35
    
    axes[1, 1].barh(x - width/2, rf_top_importance, width, label='Random Forest', color='lightgreen')
    axes[1, 1].barh(x + width/2, xgb_top_importance, width, label='XGBoost', color='lightcoral')
    axes[1, 1].set_yticks(x)
    axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in top_features])
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Performance summary table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    table_data = [
        ['Metric', 'Random Forest', 'XGBoost', 'Difference'],
        ['Accuracy', f"{rf_results['accuracy']:.3f}", f"{xgb_results['accuracy']:.3f}", 
         f"{xgb_results['accuracy'] - rf_results['accuracy']:+.3f}"],
        ['ROC-AUC', f"{rf_results['roc_auc']:.3f}", f"{xgb_results['roc_auc']:.3f}",
         f"{xgb_results['roc_auc'] - rf_results['roc_auc']:+.3f}"],
        ['Precision@10%', f"{rf_results['precision_at_10']:.3f}", f"{xgb_results['precision_at_10']:.3f}",
         f"{xgb_results['precision_at_10'] - rf_results['precision_at_10']:+.3f}"],
        ['Precision@5%', f"{rf_results['precision_at_5']:.3f}", f"{xgb_results['precision_at_5']:.3f}",
         f"{xgb_results['precision_at_5'] - rf_results['precision_at_5']:+.3f}"]
    ]
    
    table = axes[1, 2].table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 2].set_title('Performance Comparison Table', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/xgboost_vs_random_forest_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plots saved to: results/xgboost_vs_random_forest_comparison.png")

def generate_simple_shap_analysis(xgb_model, X_test, feature_names):
    """Generate simple SHAP analysis for XGBoost."""
    
    print("\n" + "="*60)
    print("GENERATING SHAP ANALYSIS FOR XGBOOST")
    print("="*60)
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values for a sample
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        print(f"Calculating SHAP values for {sample_size} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(0)
        
        # Create feature importance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SHAP-based feature importance
        sorted_idx = np.argsort(feature_importance)[-10:]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        
        ax1.barh(range(len(sorted_idx)), sorted_importance, color='skyblue')
        ax1.set_yticks(range(len(sorted_idx)))
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in sorted_features])
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('XGBoost Feature Importance (SHAP)', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # SHAP summary plot (simplified)
        sample_shap = shap_values[:10]  # First 10 samples
        sample_features = X_sample.iloc[:10]
        
        for i, feature_idx in enumerate(sorted_idx[-5:]):  # Top 5 features
            feature_name = feature_names[feature_idx]
            shap_vals = sample_shap[:, feature_idx]
            feature_vals = sample_features.iloc[:, feature_idx]
            
            colors = ['red' if val > 0 else 'blue' for val in shap_vals]
            ax2.scatter([i] * len(shap_vals), shap_vals, c=colors, alpha=0.6, s=50)
        
        ax2.set_xticks(range(5))
        ax2.set_xticklabels([feature_names[i].replace('_', ' ').title() for i in sorted_idx[-5:]], rotation=45)
        ax2.set_ylabel('SHAP Value')
        ax2.set_title('SHAP Values for Top 5 Features', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/xgboost_shap_analysis.png', dpi=300, bbox_inches='tight')
        print("SHAP analysis saved to: results/xgboost_shap_analysis.png")
        
        # Print top features
        print(f"\nTop 10 Features by SHAP Importance:")
        for i, idx in enumerate(sorted_idx[::-1], 1):
            print(f"  {i:2d}. {feature_names[idx]:25} - {feature_importance[idx]:.4f}")
            
        return feature_importance
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None

def save_champion_model(xgb_model, rf_model, xgb_results, rf_results, feature_names):
    """Save the champion model."""
    
    print("\n" + "="*60)
    print("DETERMINING CHAMPION MODEL")
    print("="*60)
    
    if xgb_results['roc_auc'] > rf_results['roc_auc']:
        champion_model = xgb_model
        champion_name = "XGBoost"
        champion_results = xgb_results
        print(f"🏆 NEW CHAMPION: XGBoost (ROC-AUC: {xgb_results['roc_auc']:.4f})")
        
        # Save XGBoost model
        joblib.dump(xgb_model, 'models/champion_xgboost.pkl')
        print("XGBoost model saved to: models/champion_xgboost.pkl")
        
    else:
        champion_model = rf_model
        champion_name = "Random Forest"
        champion_results = rf_results
        print(f"🏆 CHAMPION REMAINS: Random Forest (ROC-AUC: {rf_results['roc_auc']:.4f})")
    
    # Update feature names
    with open('models/champion_features.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    return champion_model, champion_name, champion_results

def main():
    """Main comparison pipeline."""
    print("XGBOOST VS RANDOM FOREST COMPARISON - CLEAN FEATURES")
    print("=" * 70)
    
    # Load clean data
    X, y, feature_names = load_clean_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    # Train both models
    rf_model, rf_results = train_random_forest_comparison(X_train, X_test, y_train, y_test)
    xgb_model, xgb_results = train_xgboost_clean(X_train, X_test, y_train, y_test)
    
    # Create comparison plots
    create_comparison_plots(xgb_results, rf_results, X_test, y_test, xgb_model, rf_model)
    
    # SHAP analysis for XGBoost
    feature_importance = generate_simple_shap_analysis(xgb_model, X_test, feature_names)
    
    # Determine champion
    champion_model, champion_name, champion_results = save_champion_model(
        xgb_model, rf_model, xgb_results, rf_results, feature_names
    )
    
    # Final summary
    print("\n" + "="*70)
    print("XGBOOST VS RANDOM FOREST COMPARISON COMPLETE!")
    print("="*70)
    print(f"Random Forest: ROC-AUC {rf_results['roc_auc']:.4f}, Accuracy {rf_results['accuracy']:.4f}")
    print(f"XGBoost:       ROC-AUC {xgb_results['roc_auc']:.4f}, Accuracy {xgb_results['accuracy']:.4f}")
    print(f"\n🏆 CHAMPION: {champion_name} (ROC-AUC: {champion_results['roc_auc']:.4f})")
    
    print("\nFiles generated:")
    print("  - results/xgboost_vs_random_forest_comparison.png")
    print("  - results/xgboost_shap_analysis.png")
    print("  - models/champion_*.pkl")
    
    return champion_model, champion_name, champion_results

if __name__ == "__main__":
    main()