"""
Baseline Model Implementation for Office Apocalypse Algorithm

This module implements a logistic regression baseline model for NYC office building
vacancy risk prediction. The baseline serves as a performance benchmark for advanced models.

BASELINE MODEL RATIONALE:
========================
1. INTERPRETABILITY: Logistic regression provides clear coefficient interpretation
2. SPEED: Fast training and prediction for large-scale deployment
3. BENCHMARK: Establishes minimum performance threshold for advanced models
4. FEATURE INSIGHTS: Reveals linear relationships between features and vacancy risk

MODEL ARCHITECTURE:
=================
- Logistic Regression with L2 regularization
- Feature scaling using StandardScaler  
- Temporal validation with train/val/test split
- Comprehensive evaluation metrics

EVALUATION STRATEGY:
==================
- ROC-AUC: Overall discrimination capability
- Precision@K: Actionable risk identification  
- Calibration: Probability interpretation reliability
- Feature Importance: Coefficient analysis for insights

TARGET METRIC EXPECTATIONS:
=========================
- ROC-AUC > 0.70 (good discrimination)
- Precision@10% > 0.50 (useful for targeting)
- Calibration error < 0.1 (reliable probabilities)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Custom modules
import sys
sys.path.append('src')
from temporal_validation import TemporalValidator

class BaselineModel:
    """
    Logistic Regression baseline model for vacancy prediction.
    
    Implements comprehensive evaluation and interpretation capabilities
    for establishing performance benchmarks.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize baseline model with configuration."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear',  # Good for small-medium datasets
            penalty='l2',        # L2 regularization for stability
            C=1.0               # Default regularization strength
        )
        self.calibrated_model = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'is_vacant') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for baseline model training.
        
        Args:
            data: Raw dataset with all features
            target_col: Target variable column name
            
        Returns:
            Tuple of (X_features, y_target)
        """
        # Select relevant features for baseline model
        # Focus on interpretable, engineered features
        feature_columns = [
            # Building characteristics
            'building_age', 'office_ratio', 'floor_efficiency', 'value_per_sqft',
            
            # Financial indicators  
            'land_value_ratio', 'transaction_count', 'distress_score',
            
            # Market factors
            'mta_accessibility_proxy', 'business_density_proxy', 
            'construction_activity_proxy', 'neighborhood_vacancy_risk',
            
            # Risk composites
            'economic_distress_composite', 'investment_potential_score',
            'market_competitiveness_score', 'neighborhood_vitality_index',
            
            # Location advantages
            'office_size_advantage', 'mixed_use_advantage', 'location_advantage'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in data.columns]
        missing_features = [col for col in feature_columns if col not in data.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        print(f"Using {len(available_features)} features for baseline model:")
        for i, feat in enumerate(available_features, 1):
            print(f"  {i:2d}. {feat}")
        
        # Extract features and target
        X = data[available_features].copy()
        y = data[target_col].copy()
        
        # Handle missing values more comprehensively
        print(f"\\nHandling missing values:")
        print(f"  Features with missing values: {X.isnull().sum().sum()}")
        
        # For numeric features, use median imputation
        numeric_features = X.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):  # If all values are NaN, use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
                print(f"    Imputed {col}: {X[col].isnull().sum()} remaining NaNs")
        
        # Check for any remaining NaN/inf values
        if X.isnull().sum().sum() > 0:
            print("  Warning: Still have NaN values, filling with 0")
            X = X.fillna(0)
        
        if np.isinf(X.values).sum() > 0:
            print("  Warning: Have infinite values, replacing with large finite values")
            X = X.replace([np.inf, -np.inf], [1e6, -1e6])
        
        print(f"  Final check - NaN count: {X.isnull().sum().sum()}")
        print(f"  Final check - Inf count: {np.isinf(X.values).sum()}")
        
        self.feature_names = available_features
        return X, y
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train baseline logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for calibration)
            y_val: Validation target (for calibration)
            
        Returns:
            Training results dictionary
        """
        print("\\nTRAINING BASELINE LOGISTIC REGRESSION MODEL")
        print("=" * 50)
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("Training logistic regression...")
        self.model.fit(X_train_scaled, y_train)
        
        # Train calibrated model if validation data provided
        if X_val is not None and y_val is not None:
            print("Training calibrated model...")
            X_val_scaled = self.scaler.transform(X_val)
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv='prefit'
            )
            self.calibrated_model.fit(X_val_scaled, y_val)
        
        self.is_trained = True
        
        # Training performance
        train_predictions = self.predict(X_train)
        train_probabilities = self.predict_proba(X_train)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, train_predictions),
            'precision': precision_score(y_train, train_predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_train, train_predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_train, train_predictions, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_train, train_probabilities) if len(np.unique(y_train)) > 1 else 0.5
        }
        
        print("\\nTraining Performance:")
        for metric, value in train_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions.""" 
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Use calibrated model if available, otherwise regular model
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                 dataset_name: str = "Test") -> Dict[str, Any]:
        """
        Comprehensive evaluation of baseline model.
        
        Args:
            X_test: Test features
            y_test: Test target
            dataset_name: Name for reporting
            
        Returns:
            Evaluation metrics dictionary
        """
        print(f"\\nEVALUATING BASELINE MODEL ON {dataset_name.upper()} SET")
        print("=" * 50)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Core metrics
        metrics = {
            'dataset': dataset_name,
            'n_samples': len(y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Precision@K metrics for actionable insights
        precision_at_k = self._calculate_precision_at_k(y_test, y_pred_proba, 
                                                        k_values=[0.05, 0.1, 0.2, 0.3])
        metrics.update(precision_at_k)
        
        # Print results
        print(f"\\nCore Metrics:")
        print(f"  Sample Size: {metrics['n_samples']:,}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        print(f"\\nPrecision@K (for targeting high-risk buildings):")
        for k in [0.05, 0.1, 0.2, 0.3]:
            print(f"  Precision@{k*100:2.0f}%: {metrics[f'precision_at_{k}']:.4f}")
        
        return metrics
    
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                 k_values: List[float]) -> Dict[str, float]:
        """Calculate precision@k for different thresholds."""
        results = {}
        
        for k in k_values:
            # Top k% threshold
            threshold = np.percentile(y_scores, 100 * (1 - k))
            top_k_predictions = (y_scores >= threshold).astype(int)
            
            if np.sum(top_k_predictions) > 0:
                precision_k = precision_score(y_true, top_k_predictions, zero_division=0)
            else:
                precision_k = 0.0
                
            results[f'precision_at_{k}'] = precision_k
        
        return results
    
    def plot_evaluation_results(self, X_test: pd.DataFrame, y_test: pd.Series, 
                               save_path: str = None):
        """Generate comprehensive evaluation plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Baseline Model Evaluation Results', fontsize=16, fontweight='bold')
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. ROC Curve
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            axes[0,1].plot(fpr, tpr, label=f'ROC-AUC = {auc_score:.3f}', linewidth=2)
            axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate') 
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        if len(np.unique(y_test)) > 1:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            axes[0,2].plot(recall, precision, linewidth=2)
            axes[0,2].set_xlabel('Recall')
            axes[0,2].set_ylabel('Precision')
            axes[0,2].set_title('Precision-Recall Curve')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1,0].hist(y_pred_proba[y_test==0], bins=30, alpha=0.7, label='Not Vacant', color='blue')
        axes[1,0].hist(y_pred_proba[y_test==1], bins=30, alpha=0.7, label='Vacant', color='red')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].legend()
        
        # 5. Calibration Plot
        if len(np.unique(y_test)) > 1:
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10, normalize=False
                )
                axes[1,1].plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2)
                axes[1,1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[1,1].set_xlabel('Mean Predicted Probability')
                axes[1,1].set_ylabel('Fraction of Positives')
                axes[1,1].set_title('Calibration Plot')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f'Calibration plot error:\\n{str(e)}', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
        
        # 6. Feature Importance (Coefficients)
        if self.feature_names and len(self.feature_names) <= 20:  # Only if manageable number
            coefficients = self.model.coef_[0]
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=True)
            
            # Plot top 15 features
            top_features = feature_importance.tail(15)
            colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
            axes[1,2].barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
            axes[1,2].set_yticks(range(len(top_features)))
            axes[1,2].set_yticklabels(top_features['feature'], fontsize=8)
            axes[1,2].set_xlabel('Coefficient Value')
            axes[1,2].set_title('Top 15 Feature Coefficients')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\\nEvaluation plots saved to: {save_path}")
        
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from logistic regression coefficients."""
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Model must be trained with feature names")
        
        coefficients = self.model.coef_[0]
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'importance_rank': range(1, len(coefficients) + 1)
        }).sort_values('abs_coefficient', ascending=False)
        
        importance_df['importance_rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def generate_predictions_report(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                  output_path: str = None) -> pd.DataFrame:
        """Generate detailed predictions report for analysis."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        report_df = X_test.copy()
        report_df['actual_vacant'] = y_test.values
        report_df['predicted_vacant'] = predictions
        report_df['vacancy_probability'] = probabilities
        report_df['correct_prediction'] = (predictions == y_test.values)
        
        # Risk categories based on probability
        report_df['risk_category'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low_Risk', 'Medium_Risk', 'High_Risk']
        )
        
        # Sort by probability (highest risk first)
        report_df = report_df.sort_values('vacancy_probability', ascending=False)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            print(f"\\nPredictions report saved to: {output_path}")
        
        return report_df


def run_baseline_model_pipeline():
    """Run complete baseline model pipeline."""
    print("BASELINE MODEL PIPELINE FOR OFFICE APOCALYPSE ALGORITHM")
    print("=" * 60)
    
    # Load and prepare data
    data_path = Path("data/processed/office_buildings_processed.csv")
    data = pd.read_csv(data_path)
    print(f"Loaded data: {len(data):,} records")
    
    # Create target if needed
    if 'is_vacant' not in data.columns:
        if 'target_high_vacancy_risk' in data.columns:
            data['is_vacant'] = data['target_high_vacancy_risk']
        else:
            data['is_vacant'] = (data['vacancy_risk_alert'].isin(['Orange', 'Red'])).astype(int)
        print(f"Created target variable - Vacancy rate: {data['is_vacant'].mean():.3f}")
    
    # Temporal validation split
    validator = TemporalValidator(data, target_col='is_vacant')
    train_df, val_df, test_df = validator.temporal_split()
    
    # Initialize baseline model
    baseline_model = BaselineModel(random_state=42)
    
    # Prepare features
    X_train, y_train = baseline_model.prepare_features(train_df)
    X_val, y_val = baseline_model.prepare_features(val_df)
    X_test, y_test = baseline_model.prepare_features(test_df)
    
    # Train model
    train_metrics = baseline_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    val_metrics = baseline_model.evaluate(X_val, y_val, "Validation")
    
    # Evaluate on test set
    test_metrics = baseline_model.evaluate(X_test, y_test, "Test")
    
    # Generate evaluation plots
    baseline_model.plot_evaluation_results(
        X_test, y_test, 
        save_path="results/baseline_model_evaluation.png"
    )
    
    # Feature importance analysis
    print("\\nTOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    feature_importance = baseline_model.get_feature_importance()
    for idx, row in feature_importance.head(10).iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['importance_rank']:2d}. {row['feature']:<30} {sign} {row['abs_coefficient']:.4f}")
    
    # Generate predictions report
    predictions_df = baseline_model.generate_predictions_report(
        X_test, y_test,
        output_path="results/baseline_predictions.csv"
    )
    
    # Summary
    print("\\nBASELINE MODEL SUMMARY:")
    print("=" * 30)
    print(f"Test ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print(f"Test Precision@10%: {test_metrics['precision_at_0.1']:.4f}")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    
    return baseline_model, test_metrics, feature_importance, predictions_df

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run baseline model pipeline
    model, metrics, importance, predictions = run_baseline_model_pipeline()
    
    print("\\nBASELINE MODEL PIPELINE COMPLETE!")
    print("=" * 40)
    print("Files generated:")
    print("  - results/baseline_model_evaluation.png")
    print("  - results/baseline_predictions.csv")
    print("\\nNext: Task 4.3 - Advanced Models (Random Forest, XGBoost)")