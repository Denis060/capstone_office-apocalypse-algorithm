"""
Hyperparameter Tuning Pipeline for Office Apocalypse Algorithm

This module implements comprehensive hyperparameter optimization for the
Random Forest champion model using grid search with cross-validation.

OPTIMIZATION STRATEGY:
=====================
1. GRID SEARCH: Systematic exploration of hyperparameter combinations
2. STRATIFIED CV: Maintains class balance across folds
3. SCORING METRICS: ROC-AUC (primary) + Precision@10% (business)
4. OVERFITTING PREVENTION: Validation curves and learning curves

KEY HYPERPARAMETERS:
==================
- n_estimators: Number of trees (complexity vs performance)
- max_depth: Tree depth (overfitting control)
- min_samples_split: Split threshold (generalization)
- min_samples_leaf: Leaf size (smoothing)
- max_features: Feature sampling (randomness)
- class_weight: Imbalance handling

BUSINESS FOCUS:
==============
- Primary: ROC-AUC for overall discrimination
- Secondary: Precision@10% for targeted interventions
- Interpretability: Feature importance stability
- Efficiency: Inference time for deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    make_scorer
)
import time
import joblib

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for Random Forest model.
    
    Implements grid search with cross-validation, overfitting analysis,
    and business-focused evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """Initialize hyperparameter tuner."""
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.validation_curves = {}
        
    def create_parameter_grid(self, search_type: str = 'comprehensive') -> Dict:
        """
        Create parameter grid for hyperparameter search.
        
        Args:
            search_type: 'quick', 'comprehensive', or 'fine_tune'
            
        Returns:
            Dictionary of parameter ranges
        """
        if search_type == 'quick':
            # Quick search for initial exploration
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None]
            }
        
        elif search_type == 'comprehensive':
            # Comprehensive search for best performance
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.5, None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
        elif search_type == 'fine_tune':
            # Fine-tuning around best parameters
            param_grid = {
                'n_estimators': [150, 200, 250, 300],
                'max_depth': [8, 10, 12, 15],
                'min_samples_split': [5, 8, 10, 12],
                'min_samples_leaf': [2, 3, 5, 7],
                'max_features': ['sqrt', 0.6, 0.8],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        
        else:
            raise ValueError("search_type must be 'quick', 'comprehensive', or 'fine_tune'")
        
        return param_grid
    
    def custom_precision_at_k_scorer(self, k: float = 0.1):
        """
        Create custom scorer for Precision@K metric.
        
        Args:
            k: Fraction of top predictions to evaluate
            
        Returns:
            Sklearn scorer object
        """
        def precision_at_k(y_true, y_proba):
            """Calculate precision at top k% predictions."""
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]  # Get positive class probabilities
            
            threshold = np.percentile(y_proba, 100 * (1 - k))
            top_k_pred = (y_proba >= threshold).astype(int)
            
            if np.sum(top_k_pred) == 0:
                return 0.0
            
            return precision_score(y_true, top_k_pred, zero_division=0)
        
        return make_scorer(precision_at_k, needs_proba=True)
    
    def run_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series,
                       search_type: str = 'comprehensive',
                       cv_folds: int = 5,
                       scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Run grid search with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            search_type: Type of parameter search
            cv_folds: Number of cross-validation folds
            scoring: Primary scoring metric
            
        Returns:
            Grid search results
        """
        print("\\nHYPERPARAMETER TUNING WITH GRID SEARCH")
        print("=" * 45)
        
        # Create parameter grid
        param_grid = self.create_parameter_grid(search_type)
        print(f"Search type: {search_type}")
        print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
        
        # Print parameter ranges
        print("\\nParameter Grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create base model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=1  # Use 1 job for grid search to avoid nested parallelism
        )
        
        # Set up cross-validation
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Multiple scoring metrics
        scoring_metrics = {
            'roc_auc': 'roc_auc',
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'precision_at_10': self.custom_precision_at_k_scorer(0.1)
        }
        
        # Run grid search
        print(f"\\nRunning grid search with {cv_folds}-fold CV...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring_metrics,
            refit=scoring,  # Refit on primary metric
            cv=cv_strategy,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"Grid search completed in {elapsed_time:.1f} seconds")
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # Print best results
        print(f"\\nBEST PARAMETERS:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\\nBEST CROSS-VALIDATION SCORES:")
        best_idx = grid_search.best_index_
        for metric in scoring_metrics.keys():
            score = self.cv_results.loc[best_idx, f'mean_test_{metric}']
            std = self.cv_results.loc[best_idx, f'std_test_{metric}']
            print(f"  {metric}: {score:.4f} (±{std:.4f})")
        
        return {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': self.cv_results,
            'grid_search_time': elapsed_time
        }
    
    def analyze_validation_curves(self, X_train: pd.DataFrame, y_train: pd.Series,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Generate validation curves for key hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            
        Returns:
            Validation curve results
        """
        print("\\nGENERATING VALIDATION CURVES")
        print("=" * 32)
        
        # Use best parameters as base, vary one at a time
        base_params = self.best_params.copy() if self.best_params else {
            'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,
            'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'
        }
        
        # Parameters to analyze
        param_ranges = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
            'max_depth': [3, 5, 8, 10, 12, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
            'min_samples_leaf': [1, 2, 5, 10, 15, 20],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]
        }
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        validation_results = {}
        
        for param_name, param_values in param_ranges.items():
            print(f"\\nAnalyzing {param_name}...")
            
            # Create model with base parameters
            model_params = base_params.copy()
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=1, **model_params)
            
            # Generate validation curve
            train_scores, val_scores = validation_curve(
                model, X_train, y_train,
                param_name=param_name,
                param_range=param_values,
                cv=cv_strategy,
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )
            
            validation_results[param_name] = {
                'param_values': param_values,
                'train_scores': train_scores,
                'val_scores': val_scores,
                'train_mean': train_scores.mean(axis=1),
                'train_std': train_scores.std(axis=1),
                'val_mean': val_scores.mean(axis=1),
                'val_std': val_scores.std(axis=1)
            }
            
            # Find optimal value
            best_idx = val_scores.mean(axis=1).argmax()
            best_value = param_values[best_idx]
            best_score = val_scores.mean(axis=1)[best_idx]
            
            print(f"  Best {param_name}: {best_value} (CV AUC: {best_score:.4f})")
        
        self.validation_curves = validation_results
        return validation_results
    
    def generate_learning_curves(self, X_train: pd.DataFrame, y_train: pd.Series,
                               cv_folds: int = 5) -> Dict[str, Any]:
        """
        Generate learning curves to analyze model complexity.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            
        Returns:
            Learning curve results
        """
        print("\\nGENERATING LEARNING CURVES")
        print("=" * 27)
        
        # Use best model or default parameters
        if self.best_model:
            model = self.best_model
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
                random_state=self.random_state, n_jobs=1
            )
        
        # Training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print("Computing learning curves...")
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            shuffle=True,
            random_state=self.random_state
        )
        
        learning_results = {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }
        
        # Analyze results
        final_train_score = train_scores.mean(axis=1)[-1]
        final_val_score = val_scores.mean(axis=1)[-1]
        overfitting_gap = final_train_score - final_val_score
        
        print(f"\\nLearning Curve Analysis:")
        print(f"  Final Training AUC: {final_train_score:.4f}")
        print(f"  Final Validation AUC: {final_val_score:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.05:
            print("  ⚠️ Warning: Model may be overfitting")
        else:
            print("  ✅ Overfitting appears controlled")
        
        return learning_results
    
    def create_tuning_plots(self, save_path: str = None):
        """
        Create comprehensive hyperparameter tuning visualization.
        
        Args:
            save_path: Path to save plots
        """
        print("\\nCreating hyperparameter tuning plots...")
        
        if not self.cv_results is not None:
            print("No grid search results available for plotting")
            return
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Hyperparameter Tuning Analysis - Random Forest', fontsize=16, fontweight='bold')
        
        # 1. Best parameters summary
        ax = axes[0, 0]
        if self.best_params:
            param_names = list(self.best_params.keys())
            param_values = [str(v) for v in self.best_params.values()]
            
            y_pos = np.arange(len(param_names))
            ax.barh(y_pos, [1] * len(param_names), alpha=0.7, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(param_names)
            ax.set_xlabel('Best Parameters')
            ax.set_title('Optimal Hyperparameters')
            
            # Add parameter values as text
            for i, (name, value) in enumerate(zip(param_names, param_values)):
                ax.text(0.5, i, f'{value}', ha='center', va='center', fontweight='bold')
        
        # 2. CV Score distribution
        ax = axes[0, 1]
        cv_scores = self.cv_results['mean_test_roc_auc']
        ax.hist(cv_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(cv_scores.max(), color='red', linestyle='--', linewidth=2, label=f'Best: {cv_scores.max():.4f}')
        ax.set_xlabel('Cross-Validation ROC-AUC')
        ax.set_ylabel('Frequency')
        ax.set_title('CV Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Top 10 parameter combinations
        ax = axes[0, 2]
        top_10 = self.cv_results.nlargest(10, 'mean_test_roc_auc')
        y_pos = np.arange(10)
        scores = top_10['mean_test_roc_auc'].values
        errors = top_10['std_test_roc_auc'].values
        
        ax.barh(y_pos, scores, xerr=errors, alpha=0.7, color='gold', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Rank {i+1}' for i in range(10)])
        ax.set_xlabel('CV ROC-AUC')
        ax.set_title('Top 10 Parameter Combinations')
        ax.grid(True, alpha=0.3)
        
        # 4-8. Validation curves (if available)
        validation_plots = [
            ('n_estimators', axes[1, 0]),
            ('max_depth', axes[1, 1]),
            ('min_samples_split', axes[1, 2]),
            ('min_samples_leaf', axes[2, 0]),
            ('max_features', axes[2, 1])
        ]
        
        for param_name, ax in validation_plots:
            if param_name in self.validation_curves:
                data = self.validation_curves[param_name]
                param_values = data['param_values']
                
                # Handle non-numeric values
                if param_name == 'max_features':
                    x_values = range(len(param_values))
                    x_labels = [str(v) for v in param_values]
                else:
                    x_values = param_values
                    x_labels = param_values
                
                ax.plot(x_values, data['train_mean'], 'o-', color='blue', label='Training', alpha=0.8)
                ax.fill_between(x_values, 
                               data['train_mean'] - data['train_std'],
                               data['train_mean'] + data['train_std'],
                               alpha=0.2, color='blue')
                
                ax.plot(x_values, data['val_mean'], 'o-', color='red', label='Validation', alpha=0.8)
                ax.fill_between(x_values,
                               data['val_mean'] - data['val_std'],
                               data['val_mean'] + data['val_std'],
                               alpha=0.2, color='red')
                
                ax.set_xlabel(param_name.replace('_', ' ').title())
                ax.set_ylabel('ROC-AUC')
                ax.set_title(f'Validation Curve: {param_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if param_name == 'max_features':
                    ax.set_xticks(x_values)
                    ax.set_xticklabels(x_labels, rotation=45)
            else:
                ax.text(0.5, 0.5, f'No validation curve\\nfor {param_name}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 9. Model complexity vs performance
        ax = axes[2, 2]
        if 'mean_fit_time' in self.cv_results.columns:
            fit_times = self.cv_results['mean_fit_time']
            cv_scores = self.cv_results['mean_test_roc_auc']
            
            scatter = ax.scatter(fit_times, cv_scores, alpha=0.6, c=cv_scores, cmap='viridis')
            ax.set_xlabel('Training Time (seconds)')
            ax.set_ylabel('CV ROC-AUC')
            ax.set_title('Performance vs Complexity')
            plt.colorbar(scatter, ax=ax, label='ROC-AUC')
            
            # Mark best model
            best_idx = cv_scores.idxmax()
            best_time = fit_times.iloc[best_idx]
            best_score = cv_scores.iloc[best_idx]
            ax.scatter([best_time], [best_score], color='red', s=100, marker='*', 
                      label=f'Best Model\\nTime: {best_time:.2f}s\\nAUC: {best_score:.4f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Training time\\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved tuning plots: {save_path}")
        
        plt.close()
    
    def save_results(self, save_dir: str = "results"):
        """
        Save tuning results to files.
        
        Args:
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print(f"\\nSaving hyperparameter tuning results to {save_dir}...")
        
        # Save best model
        if self.best_model:
            model_path = save_path / "best_random_forest_model.pkl"
            joblib.dump(self.best_model, model_path)
            print(f"  Saved best model: {model_path}")
        
        # Save best parameters
        if self.best_params:
            params_df = pd.DataFrame([self.best_params])
            params_path = save_path / "best_hyperparameters.csv"
            params_df.to_csv(params_path, index=False)
            print(f"  Saved best parameters: {params_path}")
        
        # Save CV results
        if self.cv_results is not None:
            cv_path = save_path / "grid_search_results.csv"
            self.cv_results.to_csv(cv_path, index=False)
            print(f"  Saved CV results: {cv_path}")
        
        # Save validation curves
        if self.validation_curves:
            for param_name, data in self.validation_curves.items():
                val_curve_df = pd.DataFrame({
                    'param_value': data['param_values'],
                    'train_mean': data['train_mean'],
                    'train_std': data['train_std'],
                    'val_mean': data['val_mean'],
                    'val_std': data['val_std']
                })
                val_path = save_path / f"validation_curve_{param_name}.csv"
                val_curve_df.to_csv(val_path, index=False)
                print(f"  Saved validation curve: {val_path}")


def run_hyperparameter_tuning():
    """Run complete hyperparameter tuning pipeline."""
    print("HYPERPARAMETER TUNING PIPELINE - OFFICE APOCALYPSE ALGORITHM")
    print("=" * 65)
    
    # Load clean dataset
    data = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Loaded clean data: {len(data):,} records")
    
    # Prepare features and target
    feature_cols = [col for col in data.columns 
                   if col not in ['target_high_vacancy_risk', 'BBL']]
    
    X = data[feature_cols].copy()
    y = data['target_high_vacancy_risk'].copy()
    
    # Clean data
    print(f"\\nCleaning data...")
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0
                X[col] = X[col].fillna(fill_val)
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts().to_dict()}")
    
    # Train/validation split for tuning
    from sklearn.model_selection import train_test_split
    X_tune, X_holdout, y_tune, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\\nTuning set: {len(X_tune):,} samples")
    print(f"Holdout set: {len(X_holdout):,} samples")
    
    # Initialize tuner
    tuner = HyperparameterTuner(random_state=42, n_jobs=-1)
    
    # Run grid search (start with comprehensive search)
    print("\\n" + "="*60)
    print("PHASE 1: COMPREHENSIVE GRID SEARCH")
    print("="*60)
    
    grid_results = tuner.run_grid_search(
        X_tune, y_tune,
        search_type='comprehensive',
        cv_folds=5,
        scoring='roc_auc'
    )
    
    # Analyze validation curves
    print("\\n" + "="*60)
    print("PHASE 2: VALIDATION CURVE ANALYSIS")
    print("="*60)
    
    validation_results = tuner.analyze_validation_curves(X_tune, y_tune, cv_folds=5)
    
    # Generate learning curves
    print("\\n" + "="*60)
    print("PHASE 3: LEARNING CURVE ANALYSIS")
    print("="*60)
    
    learning_results = tuner.generate_learning_curves(X_tune, y_tune, cv_folds=5)
    
    # Test on holdout set
    print("\\n" + "="*60)
    print("PHASE 4: HOLDOUT SET EVALUATION")
    print("="*60)
    
    best_model = tuner.best_model
    y_pred_holdout = best_model.predict(X_holdout)
    y_pred_proba_holdout = best_model.predict_proba(X_holdout)[:, 1]
    
    holdout_metrics = {
        'accuracy': accuracy_score(y_holdout, y_pred_holdout),
        'roc_auc': roc_auc_score(y_holdout, y_pred_proba_holdout),
        'precision': precision_score(y_holdout, y_pred_holdout, average='weighted'),
        'recall': recall_score(y_holdout, y_pred_holdout, average='weighted'),
        'f1': f1_score(y_holdout, y_pred_holdout, average='weighted')
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
    
    print("\\nHOLDOUT SET PERFORMANCE:")
    for metric, score in holdout_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    # Create visualizations
    tuner.create_tuning_plots("results/hyperparameter_tuning_analysis.png")
    
    # Save all results
    tuner.save_results("results")
    
    # Save holdout metrics
    holdout_df = pd.DataFrame([holdout_metrics])
    holdout_df.to_csv("results/holdout_evaluation.csv", index=False)
    
    print("\\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("="*60)
    
    print("\\n📊 FINAL RESULTS SUMMARY:")
    print(f"✅ Best CV ROC-AUC: {grid_results['best_score']:.4f}")
    print(f"✅ Holdout ROC-AUC: {holdout_metrics['roc_auc']:.4f}")
    print(f"✅ Holdout Precision@10%: {holdout_metrics['precision_at_0.1']:.4f}")
    
    print("\\n🎯 CHAMPION MODEL PARAMETERS:")
    for param, value in tuner.best_params.items():
        print(f"  {param}: {value}")
    
    print("\\n📁 Generated Files:")
    print("  - results/best_random_forest_model.pkl")
    print("  - results/best_hyperparameters.csv") 
    print("  - results/grid_search_results.csv")
    print("  - results/holdout_evaluation.csv")
    print("  - results/hyperparameter_tuning_analysis.png")
    print("\\nNext: Task 4.6 - Final Model Evaluation")
    
    return tuner, grid_results, validation_results, learning_results, holdout_metrics

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run hyperparameter tuning pipeline
    tuner, grid_results, val_results, learning_results, holdout_metrics = run_hyperparameter_tuning()