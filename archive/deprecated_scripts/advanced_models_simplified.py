"""
Advanced Models Implementation - Simplified Version

This module implements Random Forest and XGBoost models with basic explainability
for NYC office building vacancy risk prediction.
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
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
import xgboost as xgb

# Custom modules
import sys
sys.path.append('src')
from temporal_validation import TemporalValidator
from baseline_model import BaselineModel

class AdvancedModelsSimplified:
    """
    Simplified implementation of Random Forest and XGBoost.
    
    Focuses on model training, evaluation, and comparison with robust error handling.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize advanced models with configuration."""
        self.random_state = random_state
        self.models = {}
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
                    scale_pos_weight=1.0
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
        
        print(f"\\nUsing {len(available_features)} features for advanced models")
        
        # Extract features and target
        X = data[available_features].copy()
        y = data[target_col].copy()
        
        # Handle missing values
        numeric_features = X.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if X[col].isnull().sum() > 0:
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
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Extract feature importance from trained models.
        
        Returns:
            Dictionary of feature importance dataframes by model
        """
        importance_results = {}
        
        for model_key, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_results[model_key] = importance_df
                
                print(f"\\n{self.model_configs[model_key]['name']} - Top 10 Features:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_results
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """
        Create comprehensive evaluation plots.
        
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save plots
        """
        print("\\nCreating evaluation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Models Evaluation - Office Vacancy Prediction', fontsize=16, fontweight='bold')
        
        # Colors for models
        colors = {'random_forest': 'green', 'xgboost': 'orange'}
        
        # 1. ROC Curves
        ax = axes[0, 0]
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, color=colors[model_key], linewidth=2,
                   label=f'{self.model_configs[model_key]["name"]} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax = axes[0, 1]
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            ax.plot(recall, precision, color=colors[model_key], linewidth=2,
                   label=f'{self.model_configs[model_key]["name"]}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Feature Importance Comparison
        ax = axes[0, 2]
        importance_results = self.get_feature_importance()
        
        if len(importance_results) >= 2:
            # Compare top 10 features from both models
            rf_imp = importance_results.get('random_forest', pd.DataFrame())
            xgb_imp = importance_results.get('xgboost', pd.DataFrame())
            
            if not rf_imp.empty and not xgb_imp.empty:
                # Get common top features
                top_features = list(set(rf_imp.head(10)['feature'].tolist() + xgb_imp.head(10)['feature'].tolist()))[:10]
                
                rf_values = []
                xgb_values = []
                for feature in top_features:
                    rf_val = rf_imp[rf_imp['feature'] == feature]['importance'].values
                    xgb_val = xgb_imp[xgb_imp['feature'] == feature]['importance'].values
                    rf_values.append(rf_val[0] if len(rf_val) > 0 else 0)
                    xgb_values.append(xgb_val[0] if len(xgb_val) > 0 else 0)
                
                x_pos = np.arange(len(top_features))
                ax.barh(x_pos - 0.2, rf_values, 0.4, label='Random Forest', color='green', alpha=0.7)
                ax.barh(x_pos + 0.2, xgb_values, 0.4, label='XGBoost', color='orange', alpha=0.7)
                
                ax.set_yticks(x_pos)
                ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in top_features], fontsize=8)
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance Comparison')
                ax.legend()
        
        # 4. Confusion Matrix - Random Forest
        ax = axes[1, 0]
        if 'random_forest' in self.models:
            y_pred = self.models['random_forest'].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title('Random Forest - Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # 5. Confusion Matrix - XGBoost
        ax = axes[1, 1]
        if 'xgboost' in self.models:
            y_pred = self.models['xgboost'].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Oranges')
            ax.set_title('XGBoost - Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # 6. Model Performance Comparison
        ax = axes[1, 2]
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        metrics = ['roc_auc', 'accuracy', 'precision_at_0.1']
        metric_labels = ['ROC-AUC', 'Accuracy', 'Precision@10%']
        
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_key, results in evaluation_results.items():
            model_names.append(self.model_configs[model_key]['name'])
            for metric in metrics:
                metric_values[metric].append(results[metric])
        
        x_pos = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax.bar(x_pos + i * width, metric_values[metric], width, 
                  label=metric_labels[i], alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved evaluation plots: {save_path}")
        
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
    """Run complete advanced models pipeline."""
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
    advanced_models = AdvancedModelsSimplified(random_state=42)
    
    # Prepare features
    X_train, y_train = advanced_models.prepare_features(train_df)
    X_val, y_val = advanced_models.prepare_features(val_df)
    X_test, y_test = advanced_models.prepare_features(test_df)
    
    # Train advanced models
    training_results = advanced_models.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    evaluation_results = advanced_models.evaluate_models(X_test, y_test)
    
    # Get feature importance
    importance_results = advanced_models.get_feature_importance()
    
    # Create evaluation plots
    advanced_models.create_evaluation_plots(X_test, y_test, "results/advanced_models_evaluation.png")
    
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
    
    # Save feature importance
    for model_key, importance_df in importance_results.items():
        importance_df.to_csv(f"results/{model_key}_feature_importance.csv", index=False)
    
    print("\\nADVANCED MODELS PIPELINE COMPLETE!")
    print("=" * 45)
    print("Generated files:")
    print("  - results/advanced_models_evaluation.png")
    print("  - results/model_comparison.csv")
    print("  - results/random_forest_feature_importance.csv")
    print("  - results/xgboost_feature_importance.csv")
    print("\\nNext: Task 4.4 - Hyperparameter Tuning")
    
    return advanced_models, evaluation_results, importance_results, comparison_df

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run advanced models pipeline
    models, results, importance, comparison = run_advanced_models_pipeline()