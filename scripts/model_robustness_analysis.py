"""
Model Robustness Analysis: Investigating High Accuracy and Adding XGBoost
Office Apocalypse Algorithm - Professor Meeting Analysis

This script addresses concerns about 99.99% accuracy by:
1. Checking for data leakage and overfitting
2. Adding XGBoost for model comparison
3. Implementing more robust validation
4. Analyzing feature importance for suspicious patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, cross_validate, TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost installed and available")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def investigate_high_accuracy():
    """
    Investigate why the model is achieving 99.99% accuracy
    """
    print("🔍 INVESTIGATING HIGH ACCURACY CONCERNS")
    print("=" * 60)
    
    # Load data
    features_path = Path("../data/features/office_features_cross_dataset_integrated.csv")
    
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return None
    
    df = pd.read_csv(features_path)
    print(f"✅ Loaded data: {len(df):,} buildings, {len(df.columns)} features")
    
    # Create target variable (same as in notebook)
    if 'vacancy_risk_early_warning' in df.columns:
        high_risk_threshold = df['vacancy_risk_early_warning'].quantile(0.8)
        y = (df['vacancy_risk_early_warning'] > high_risk_threshold).astype(int)
        print(f"📊 Target created: {y.sum():,} high-risk buildings ({y.mean()*100:.1f}%)")
    else:
        print("❌ Cannot create target variable")
        return None
    
    # Check for potential data leakage
    print(f"\n🚨 CHECKING FOR DATA LEAKAGE:")
    
    # Look for features that might be leaking future information
    suspicious_features = []
    feature_cols = [col for col in df.columns if col not in ['BBL', 'vacancy_risk_early_warning']]
    
    for col in feature_cols:
        if 'vacancy' in col.lower() or 'risk' in col.lower():
            if col != 'vacancy_risk_early_warning':
                suspicious_features.append(col)
    
    print(f"   • Suspicious features (contain 'vacancy' or 'risk'): {len(suspicious_features)}")
    for feat in suspicious_features[:10]:  # Show first 10
        print(f"     - {feat}")
    if len(suspicious_features) > 10:
        print(f"     ... and {len(suspicious_features) - 10} more")
    
    # Check correlation with target
    X = df[feature_cols].select_dtypes(include=[np.number])
    print(f"\n📈 HIGH CORRELATION FEATURES (>0.9 with target):")
    
    correlations = []
    for col in X.columns:
        if not X[col].isna().all():
            corr = abs(np.corrcoef(X[col].fillna(0), y)[0, 1])
            if corr > 0.9:
                correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    if correlations:
        print(f"   • Found {len(correlations)} features with >90% correlation:")
        for feat, corr in correlations[:10]:
            print(f"     - {feat}: {corr:.4f}")
    else:
        print(f"   • No features with >90% correlation found")
    
    return df, X, y, suspicious_features

def train_models_with_xgboost(X, y):
    """
    Train multiple models including XGBoost with robust validation
    """
    print(f"\n🤖 TRAINING MODELS WITH XGBOOST")
    print("=" * 50)
    
    # Prepare data
    X_clean = X.fillna(0)  # Handle missing values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   • Training set: {X_train.shape[0]:,} samples")
    print(f"   • Test set: {X_test.shape[0]:,} samples")
    print(f"   • Features: {X_train.shape[1]}")
    print(f"   • Class balance: {y_train.mean():.3f} positive rate")
    
    # Define models including XGBoost
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        print(f"   ✅ Added XGBoost to model comparison")
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use scaled data for logistic regression, original for tree-based
        if 'Logistic' in name:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Train model
        model.fit(X_train_model, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }
        
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1: {f1:.4f}")
        print(f"   • ROC-AUC: {roc_auc:.4f}")
    
    return results, scaler, X_train, X_test, y_train, y_test

def robust_cross_validation(X, y):
    """
    Perform more robust cross-validation to check for overfitting
    """
    print(f"\n🔄 ROBUST CROSS-VALIDATION ANALYSIS")
    print("=" * 50)
    
    X_clean = X.fillna(0)
    
    # Multiple CV strategies
    cv_strategies = {
        'StratifiedKFold (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'StratifiedKFold (10)': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    }
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    
    cv_results = {}
    
    for cv_name, cv_strategy in cv_strategies.items():
        print(f"\n📊 {cv_name} Results:")
        cv_results[cv_name] = {}
        
        for model_name, model in models.items():
            # Perform cross-validation
            scores = cross_validate(
                model, X_clean, y,
                cv=cv_strategy,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                return_train_score=True
            )
            
            # Calculate means and stds
            test_acc = scores['test_accuracy'].mean()
            train_acc = scores['train_accuracy'].mean()
            test_auc = scores['test_roc_auc'].mean()
            train_auc = scores['train_roc_auc'].mean()
            
            cv_results[cv_name][model_name] = {
                'test_accuracy': test_acc,
                'train_accuracy': train_acc,
                'test_auc': test_auc,
                'train_auc': train_auc,
                'overfitting_acc': train_acc - test_acc,
                'overfitting_auc': train_auc - test_auc
            }
            
            print(f"   {model_name}:")
            print(f"     • Test Accuracy: {test_acc:.4f} (±{scores['test_accuracy'].std():.4f})")
            print(f"     • Train Accuracy: {train_acc:.4f}")
            print(f"     • Overfitting (Acc): {train_acc - test_acc:.4f}")
            print(f"     • Test ROC-AUC: {test_auc:.4f} (±{scores['test_roc_auc'].std():.4f})")
            print(f"     • Train ROC-AUC: {train_auc:.4f}")
            print(f"     • Overfitting (AUC): {train_auc - test_auc:.4f}")
    
    return cv_results

def analyze_feature_importance(results, X, feature_names):
    """
    Analyze feature importance to understand model predictions
    """
    print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Get best model (by ROC-AUC)
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"   • Analyzing {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importance = best_model.feature_importances_
        importance_type = "Tree-based importance"
    elif hasattr(best_model, 'coef_'):
        # Linear models
        importance = abs(best_model.coef_[0])
        importance_type = "Coefficient magnitude"
    else:
        print("   ⚠️ Cannot extract feature importance from this model")
        return
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"   • Using {importance_type}")
    print(f"\n📈 Top 15 Most Important Features:")
    
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Check for suspicious patterns
    top_features = importance_df.head(10)['feature'].tolist()
    suspicious_top_features = [f for f in top_features if 'vacancy' in f.lower() or 'risk' in f.lower()]
    
    if suspicious_top_features:
        print(f"\n🚨 SUSPICIOUS TOP FEATURES (potential leakage):")
        for feat in suspicious_top_features:
            print(f"   • {feat}")
    else:
        print(f"\n✅ No obviously suspicious features in top 10")
    
    return importance_df

def create_model_comparison_plot(results):
    """
    Create visualization comparing model performance
    """
    print(f"\n📊 CREATING MODEL COMPARISON VISUALIZATION")
    
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        ax = axes[i]
        bars = ax.bar(models, values, alpha=0.7)
        ax.set_title(f'{metric.upper().replace("_", "-")}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal line at 0.95 to show "suspiciously high" threshold
        if metric in ['accuracy', 'roc_auc']:
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, 
                      label='Suspiciously High (>95%)')
            ax.legend()
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.suptitle('Model Performance Comparison - Investigating High Accuracy', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Save plot
    plot_path = Path("../results/model_performance/model_robustness_analysis.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved plot: {plot_path}")
    
    plt.show()

def main():
    """
    Main analysis function
    """
    print("🔍 MODEL ROBUSTNESS ANALYSIS FOR PROFESSOR MEETING")
    print("=" * 70)
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Purpose: Investigate 99.99% accuracy and add XGBoost comparison")
    print()
    
    # 1. Investigate high accuracy
    data_result = investigate_high_accuracy()
    if data_result is None:
        return
    
    df, X, y, suspicious_features = data_result
    
    # 2. Train models with XGBoost
    results, scaler, X_train, X_test, y_train, y_test = train_models_with_xgboost(X, y)
    
    # 3. Robust cross-validation
    cv_results = robust_cross_validation(X, y)
    
    # 4. Feature importance analysis
    importance_df = analyze_feature_importance(results, X, X.columns.tolist())
    
    # 5. Create comparison visualization
    create_model_comparison_plot(results)
    
    # 6. Summary and recommendations
    print(f"\n📋 SUMMARY AND RECOMMENDATIONS FOR PROFESSOR")
    print("=" * 60)
    
    # Find highest performing model
    best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    best_auc = results[best_model]['roc_auc']
    best_acc = results[best_model]['accuracy']
    
    print(f"   🏆 Best Model: {best_model}")
    print(f"   📊 Performance: {best_acc:.1%} accuracy, {best_auc:.4f} ROC-AUC")
    
    # Check if still suspiciously high
    if best_auc > 0.98 or best_acc > 0.95:
        print(f"\n   🚨 STILL SUSPICIOUSLY HIGH PERFORMANCE:")
        print(f"      • This suggests potential data leakage or overfitting")
        print(f"      • Recommend further investigation of feature engineering")
        print(f"      • Consider removing features with 'vacancy' or 'risk' in name")
    else:
        print(f"\n   ✅ REASONABLE PERFORMANCE ACHIEVED")
    
    # XGBoost comparison
    if 'XGBoost' in results:
        xgb_auc = results['XGBoost']['roc_auc']
        xgb_acc = results['XGBoost']['accuracy']
        print(f"\n   🚀 XGBoost Performance: {xgb_acc:.1%} accuracy, {xgb_auc:.4f} ROC-AUC")
        
        if abs(xgb_auc - best_auc) < 0.01:
            print(f"      • Similar performance to other models (good consistency)")
        else:
            print(f"      • Different performance - suggests model-specific effects")
    
    # Overfitting analysis
    print(f"\n   🔄 Overfitting Analysis (from 5-fold CV):")
    if 'StratifiedKFold (5)' in cv_results:
        for model_name, cv_result in cv_results['StratifiedKFold (5)'].items():
            overfitting_auc = cv_result['overfitting_auc']
            if overfitting_auc > 0.05:
                print(f"      • {model_name}: HIGH overfitting ({overfitting_auc:.3f})")
            elif overfitting_auc > 0.02:
                print(f"      • {model_name}: Moderate overfitting ({overfitting_auc:.3f})")
            else:
                print(f"      • {model_name}: Low overfitting ({overfitting_auc:.3f})")
    
    print(f"\n   💡 RECOMMENDATIONS:")
    print(f"      1. Review feature engineering for potential target leakage")
    print(f"      2. Consider using simpler models if accuracy remains >98%")
    print(f"      3. Validate results on completely held-out test set")
    print(f"      4. Consider temporal validation (if time component exists)")
    
    # Save results
    results_path = Path("../results/model_performance/robustness_analysis_results.txt")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("MODEL ROBUSTNESS ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n\n")
        
        if suspicious_features:
            f.write("SUSPICIOUS FEATURES (potential leakage):\n")
            for feat in suspicious_features:
                f.write(f"  - {feat}\n")
    
    print(f"\n   ✅ Results saved to: {results_path}")
    print(f"\n🎯 Analysis complete! Ready for professor meeting.")

if __name__ == "__main__":
    main()