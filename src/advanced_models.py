"""
Advanced Models Implementation for Office Apocalypse Algorithm

This module implements Random Forest and XGBoost models with SHAP explainability
for NYC office building vacancy risk prediction.

ADVANCED MODEL STRATEGY:
======================
1. RANDOM FOREST: Ensemble learning with feature importance
2. XGBOOST: Gradient boosting with advanced regularization
3. SHAP INTEGRATION: Model-agnostic explainability
4. HYPERPARAMETER OPTIMIZATION: Grid search for best performance

MODEL COMPARISON FRAMEWORK:
=========================
- Baseline: Logistic Regression (linear relationships)
- Random Forest: Non-linear patterns, feature interactions
- XGBoost: Advanced boosting, missing value handling, regularization
- Evaluation: ROC-AUC, Precision@K, calibration, feature importance

EXPLAINABILITY FOCUS:
===================
- SHAP values for individual predictions
- Feature importance rankings
- Interaction effects analysis
- Policy-relevant insights for stakeholders

TARGET PERFORMANCE:
=================
- Exceed baseline ROC-AUC (currently 0.9993)
- Maintain high Precision@10% for targeting
- Provide interpretable explanations
- Fast inference for operational deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import xgboost as xgb

# Explainability
import shap

# Custom modules
import sys
sys.path.append('src')
from temporal_validation import TemporalValidator
from baseline_model import BaselineModel

class AdvancedModels:
    """
    Implementation of Random Forest and XGBoost with SHAP explainability.
    
    Provides comprehensive comparison against baseline logistic regression
    with focus on interpretability and operational deployment.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize advanced models with configuration."""
        self.random_state = random_state
        self.models = {}
        self.shap_explainers = {}
        self.feature_names = None
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'name': 'Random Forest'
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    scale_pos_weight=1.0  # Adjust for class imbalance if needed
                ),
                'name': 'XGBoost'
            }
        }
    
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'is_vacant') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare comprehensive feature set for advanced models.
        
        Args:
            data: Raw dataset
            target_col: Target variable column
            
        Returns:
            Tuple of (X_features, y_target)
        """
        # Comprehensive feature set for tree-based models
        feature_columns = [
            # Building characteristics
            'building_age', 'office_ratio', 'floor_efficiency', 'value_per_sqft',
            'land_value_ratio', 'lotarea', 'bldgarea', 'officearea', 'numfloors',
            
            # Financial indicators
            'transaction_count', 'deed_count', 'mortgage_count', 'distress_score',
            'assessland', 'assesstot', 'yearbuilt',
            
            # Market factors
            'mta_accessibility_proxy', 'business_density_proxy', 
            'construction_activity_proxy', 'neighborhood_vacancy_risk',
            
            # Risk composites
            'economic_distress_composite', 'investment_potential_score',
            'market_competitiveness_score', 'neighborhood_vitality_index',
            'modernization_potential', 'building_quality',
            
            # Location advantages
            'office_size_advantage', 'mixed_use_advantage', 'location_advantage'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in data.columns]
        missing_features = [col for col in feature_columns if col not in data.columns]
        
        if missing_features:
            print(f"Note: {len(missing_features)} features not available in dataset")
        
        print(f"\\nUsing {len(available_features)} features for advanced models:")
        
        # Extract features and target
        X = data[available_features].copy()
        y = data[target_col].copy()
        
        # Advanced preprocessing for tree-based models
        # Tree models handle missing values better, but clean data is still preferred
        
        # Handle missing values
        numeric_features = X.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if X[col].isnull().sum() > 0:
                # Use median imputation
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()])
        X = X.fillna(0)
        
        # Create additional engineered features for tree models
        if 'building_age' in X.columns and 'value_per_sqft' in X.columns:
            X['age_value_interaction'] = X['building_age'] * X['value_per_sqft']
        
        if 'office_ratio' in X.columns and 'floor_efficiency' in X.columns:
            X['office_efficiency_interaction'] = X['office_ratio'] * X['floor_efficiency']
        
        # Feature scaling not needed for tree-based models
        print(f"Final feature matrix: {X.shape}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train Random Forest and XGBoost models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping in XGBoost)
            y_val: Validation target
            
        Returns:
            Training results dictionary
        """
        print("\\nTRAINING ADVANCED MODELS")
        print("=" * 40)
        
        training_results = {}
        
        for model_key, config in self.model_configs.items():
            print(f"\\nTraining {config['name']}...")
            
            model = config['model']
            
            # Special handling for XGBoost with validation
            if model_key == 'xgboost' and X_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            self.models[model_key] = model
            
            # Training performance
            train_pred = model.predict(X_train)
            train_proba = model.predict_proba(X_train)[:, 1]
            
            train_metrics = {
                'accuracy': accuracy_score(y_train, train_pred),
                'precision': precision_score(y_train, train_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_train, train_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_train, train_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y_train, train_proba)
            }
            
            training_results[model_key] = train_metrics
            
            print(f"  Training ROC-AUC: {train_metrics['roc_auc']:.4f}")
            print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
        
        self.is_trained = True
        print(f"\\nTrained {len(self.models)} advanced models successfully!")
        return training_results
    
    def initialize_shap_explainers(self, X_sample: pd.DataFrame, max_samples: int = 100):
        """
        Initialize SHAP explainers for model interpretability.
        
        Args:
            X_sample: Sample data for explainer initialization
            max_samples: Maximum samples for SHAP background
        """
        print("\\nInitializing SHAP explainers...")
        
        # Use subset for SHAP background to speed up computation
        background_data = X_sample.sample(min(max_samples, len(X_sample)), random_state=self.random_state)
        
        for model_key, model in self.models.items():
            print(f"  Setting up SHAP for {self.model_configs[model_key]['name']}...")
            
            if model_key == 'random_forest':
                # Tree explainer for Random Forest
                self.shap_explainers[model_key] = shap.TreeExplainer(model)
            elif model_key == 'xgboost':
                # Tree explainer for XGBoost
                self.shap_explainers[model_key] = shap.TreeExplainer(model)
        
        print("SHAP explainers initialized!")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation results by model
        """
        print("\\nEVALUATING ADVANCED MODELS")
        print("=" * 35)
        
        evaluation_results = {}
        
        for model_key, model in self.models.items():
            model_name = self.model_configs[model_key]['name']
            print(f"\\nEvaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Core metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Precision@K for operational targeting
            for k in [0.05, 0.1, 0.2, 0.3]:
                threshold = np.percentile(y_pred_proba, 100 * (1 - k))
                top_k_pred = (y_pred_proba >= threshold).astype(int)
                if np.sum(top_k_pred) > 0:
                    precision_k = precision_score(y_test, top_k_pred, zero_division=0)
                else:
                    precision_k = 0.0
                metrics[f'precision_at_{k}'] = precision_k
            
            evaluation_results[model_key] = metrics
            
            # Print results
            print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
            print(f"  Accuracy:        {metrics['accuracy']:.4f}")
            print(f"  Precision@10%:   {metrics['precision_at_0.1']:.4f}")
        
        return evaluation_results
    
    def generate_shap_explanations(self, X_sample: pd.DataFrame, save_path: str = None) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model interpretability.
        
        Args:
            X_sample: Sample data for explanation
            save_path: Path to save SHAP plots
            
        Returns:
            SHAP values and analysis
        """
        print("\\nGENERATING SHAP EXPLANATIONS")
        print("=" * 35)
        
        shap_results = {}
        
        # Limit sample size for performance
        sample_size = min(100, len(X_sample))
        X_explain = X_sample.sample(sample_size, random_state=self.random_state)
        
        for model_key, explainer in self.shap_explainers.items():
            model_name = self.model_configs[model_key]['name']
            print(f"\\nGenerating SHAP values for {model_name}...")
            
            try:
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_explain)
                
                # For binary classification, take positive class SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                
                shap_results[model_key] = {
                    'shap_values': shap_values,
                    'feature_names': self.feature_names,
                    'sample_data': X_explain
                }
                
                # Feature importance from SHAP
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
                }).sort_values('mean_abs_shap', ascending=False)
                
                shap_results[model_key]['feature_importance'] = feature_importance
                
                print(f"  Generated SHAP values: {shap_values.shape}")
                print(f"  Top 5 important features:")
                for idx, row in feature_importance.head(5).iterrows():
                    print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
                
            except Exception as e:
                print(f"  Error generating SHAP for {model_name}: {str(e)}")
                continue
        
        # Generate SHAP plots
        if save_path:
            self._create_shap_plots(shap_results, save_path)
        
        return shap_results
    
    def _create_shap_plots(self, shap_results: Dict, base_path: str):
        """Create comprehensive SHAP visualization plots."""
        print("\\nCreating SHAP visualizations...")
        
        for model_key, results in shap_results.items():
            if 'shap_values' not in results:
                continue
                
            model_name = self.model_configs[model_key]['name']
            shap_values = results['shap_values']
            X_data = results['sample_data']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'SHAP Analysis - {model_name}', fontsize=16, fontweight='bold')
            
            # 1. Summary plot (beeswarm)
            try:
                plt.sca(axes[0,0])
                shap.summary_plot(shap_values, X_data, plot_type="dot", show=False, max_display=15)
                axes[0,0].set_title('SHAP Summary Plot (Feature Impact)')
            except Exception as e:
                axes[0,0].text(0.5, 0.5, f'Summary plot error: {str(e)[:50]}...', 
                              ha='center', va='center', transform=axes[0,0].transAxes)
            
            # 2. Feature importance bar plot
            if 'feature_importance' in results:
                feature_importance = results['feature_importance']
                top_features = feature_importance.head(15)
                axes[0,1].barh(range(len(top_features)), top_features['mean_abs_shap'], 
                              color='skyblue', alpha=0.7)
                axes[0,1].set_yticks(range(len(top_features)))
                axes[0,1].set_yticklabels(top_features['feature'], fontsize=8)
                axes[0,1].set_xlabel('Mean |SHAP Value|')
                axes[0,1].set_title('Feature Importance (SHAP)')
                axes[0,1].grid(True, alpha=0.3)
            else:
                # Create manual feature importance from SHAP values
                feature_importance_vals = np.abs(shap_values).mean(axis=0)
                feature_names = list(X_data.columns)
                sorted_idx = np.argsort(feature_importance_vals)[::-1][:15]
                
                axes[0,1].barh(range(len(sorted_idx)), feature_importance_vals[sorted_idx], 
                              color='skyblue', alpha=0.7)
                axes[0,1].set_yticks(range(len(sorted_idx)))
                axes[0,1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
                axes[0,1].set_xlabel('Mean |SHAP Value|')
                axes[0,1].set_title('Feature Importance (SHAP)')
                axes[0,1].grid(True, alpha=0.3)
            
            # 3. Waterfall plot for first sample
            try:
                plt.sca(axes[1,0])
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values[0], 
                    base_values=np.mean(shap_values), 
                    data=X_data.iloc[0].values,
                    feature_names=list(X_data.columns)
                ), show=False)
                axes[1,0].set_title('SHAP Waterfall (First Sample)')
            except:
                axes[1,0].text(0.5, 0.5, 'Waterfall plot not available', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
            
            # 4. Force plot alternative - just show top contributors
            if len(shap_values) > 0:
                sample_shap = shap_values[0]
                sample_data = X_data.iloc[0]
                
                # Get top 10 contributors
                contrib_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'shap_value': sample_shap,
                    'feature_value': sample_data.values
                })
                contrib_df['abs_shap'] = np.abs(contrib_df['shap_value'])
                top_contrib = contrib_df.nlargest(10, 'abs_shap')
                
                colors = ['red' if x < 0 else 'blue' for x in top_contrib['shap_value']]
                axes[1,1].barh(range(len(top_contrib)), top_contrib['shap_value'], 
                              color=colors, alpha=0.7)
                axes[1,1].set_yticks(range(len(top_contrib)))
                axes[1,1].set_yticklabels(top_contrib['feature'], fontsize=8)
                axes[1,1].set_xlabel('SHAP Value')
                axes[1,1].set_title('Top Contributors (First Sample)')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{base_path}_{model_key}_shap_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved SHAP plots for {model_name}: {plot_path}")
            plt.close()
    
    def compare_models(self, evaluation_results: Dict[str, Dict], baseline_metrics: Dict = None):
        """
        Generate comprehensive model comparison analysis.
        
        Args:
            evaluation_results: Results from evaluate_models()
            baseline_metrics: Baseline model metrics for comparison
        """
        print("\\nMODEL COMPARISON ANALYSIS")
        print("=" * 35)
        
        # Create comparison dataframe
        comparison_data = []
        
        # Add baseline if provided
        if baseline_metrics:
            baseline_row = {
                'Model': 'Logistic Regression (Baseline)',
                'ROC-AUC': baseline_metrics.get('roc_auc', 0),
                'Accuracy': baseline_metrics.get('accuracy', 0),
                'Precision@10%': baseline_metrics.get('precision_at_0.1', 0),
                'F1-Score': baseline_metrics.get('f1', 0)
            }
            comparison_data.append(baseline_row)
        
        # Add advanced models
        for model_key, metrics in evaluation_results.items():
            row = {
                'Model': metrics['model_name'],
                'ROC-AUC': metrics['roc_auc'],
                'Accuracy': metrics['accuracy'], 
                'Precision@10%': metrics['precision_at_0.1'],
                'F1-Score': metrics['f1']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print("\\nModel Performance Comparison:")
        print("-" * 60)
        for _, row in comparison_df.iterrows():
            print(f"{row['Model']:<25} | ROC-AUC: {row['ROC-AUC']:.4f} | "
                  f"Acc: {row['Accuracy']:.4f} | P@10%: {row['Precision@10%']:.4f}")
        
        # Determine champion model
        best_model = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
        print(f"\\nCHAMPION MODEL: {best_model['Model']}")
        print(f"  ROC-AUC: {best_model['ROC-AUC']:.4f}")
        print(f"  Precision@10%: {best_model['Precision@10%']:.4f}")
        
        return comparison_df, best_model


def run_advanced_models_pipeline():
    """Run complete advanced models pipeline with SHAP explainability."""
    print("ADVANCED MODELS PIPELINE - OFFICE APOCALYPSE ALGORITHM")
    print("=" * 65)
    
    # Load processed data
    data_path = Path("data/processed/office_buildings_processed.csv")
    data = pd.read_csv(data_path)
    print(f"Loaded data: {len(data):,} records")
    
    # Create target if needed
    if 'is_vacant' not in data.columns:
        if 'target_high_vacancy_risk' in data.columns:
            data['is_vacant'] = data['target_high_vacancy_risk']
        else:
            data['is_vacant'] = (data['vacancy_risk_alert'].isin(['Orange', 'Red'])).astype(int)
        print(f"Target variable - Vacancy rate: {data['is_vacant'].mean():.3f}")
    
    # Temporal validation split
    validator = TemporalValidator(data, target_col='is_vacant')
    train_df, val_df, test_df = validator.temporal_split()
    
    # Initialize advanced models
    advanced_models = AdvancedModels(random_state=42)
    
    # Prepare features
    X_train, y_train = advanced_models.prepare_features(train_df)
    X_val, y_val = advanced_models.prepare_features(val_df)
    X_test, y_test = advanced_models.prepare_features(test_df)
    
    # Train advanced models
    training_results = advanced_models.train_models(X_train, y_train, X_val, y_val)
    
    # Initialize SHAP explainers
    advanced_models.initialize_shap_explainers(X_train, max_samples=100)
    
    # Evaluate models
    evaluation_results = advanced_models.evaluate_models(X_test, y_test)
    
    # Generate SHAP explanations
    shap_results = advanced_models.generate_shap_explanations(
        X_test, save_path="results/advanced_models"
    )
    
    # Load baseline results for comparison
    baseline_model = BaselineModel()
    X_train_baseline, _ = baseline_model.prepare_features(train_df)
    X_test_baseline, _ = baseline_model.prepare_features(test_df)
    baseline_model.train(X_train_baseline, y_train)
    baseline_metrics = baseline_model.evaluate(X_test_baseline, y_test, "Test")
    
    # Model comparison
    comparison_df, champion_model = advanced_models.compare_models(
        evaluation_results, baseline_metrics
    )
    
    # Save results
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    
    print("\\nADVANCED MODELS PIPELINE COMPLETE!")
    print("=" * 45)
    print("Generated files:")
    print("  - results/advanced_models_*_shap_analysis.png")
    print("  - results/model_comparison.csv")
    print("\\nNext: Task 4.4 - Hyperparameter Tuning")
    
    return advanced_models, evaluation_results, shap_results, comparison_df

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run advanced models pipeline
    models, results, shap, comparison = run_advanced_models_pipeline()