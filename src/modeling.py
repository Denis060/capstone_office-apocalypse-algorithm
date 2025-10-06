"""
Modeling Module for Office Apocalypse Algorithm

This module implements machine learning models to predict NYC office building
vacancy risk using the engineered features.

MODEL STRATEGY:
==============
1. BASELINE MODEL: Logistic Regression
   - Interpretable coefficients
   - Fast training and prediction
   - Good for understanding feature relationships

2. COMPARISON MODEL: Random Forest
   - Handles non-linear relationships
   - Feature importance analysis
   - Robust to outliers and missing data

3. EVALUATION METRICS:
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC for probability calibration
   - Confusion Matrix for error analysis

FEATURE SELECTION:
================
- All 23 engineered features used initially
- Feature importance analysis to identify key predictors
- Correlation analysis to detect multicollinearity

TARGET VARIABLE:
==============
Binary classification: 0 = Occupied, 1 = Vacant
Note: Current implementation uses synthetic labels.
Real implementation would use actual vacancy data.

VALIDATION STRATEGY:
==================
- Train/Validation/Test split (70/15/15)
- Stratified sampling to maintain class balance
- Cross-validation for robust performance estimates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_features(features_dir: str = "data/features", x_filename: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load engineered features and target variable.

    Args:
        features_dir: Directory containing feature files

    Returns:
        Tuple of (X_features, y_target)
    """
    features_path = Path(features_dir)

    # Load features
    # Allow caller to specify an alternate X file (avoid overwriting canonical X_features.csv)
    X_path = features_path / (x_filename if x_filename is not None else "X_features.csv")
    y_path = features_path / "y_target.csv"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Feature files not found in {features_path}")

    print(f"Loading features from {features_path}...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]  # Get first column as Series

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def prepare_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets and scale features.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Data cleaning: Handle infinity and NaN values
    print("Cleaning data: handling infinity and NaN values...")
    X = X.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    X = X.fillna(0)  # Fill NaN with 0 (reasonable for our features)

    # Verify no more infinity values
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        print(f"Warning: Still {inf_count} infinity values after cleaning")

    # First split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                            random_state: int = 42) -> LogisticRegression:
    """
    Train Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed

    Returns:
        Trained LogisticRegression model
    """
    print("Training Logistic Regression model...")

    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train, y_train)
    print("Logistic Regression training complete")

    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier model
    """
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)
    print("Random Forest training complete")

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                  model_name: str) -> Dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Print results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

    return metrics


def analyze_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """
    Analyze and return feature importance for the model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        model_name: Name of the model

    Returns:
        DataFrame with feature importance rankings
    """
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance_scores = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importance_scores = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Model {model_name} doesn't have feature importance attribute")

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\nTop 10 features for {model_name}:")
    for i, row in importance_df.head(10).iterrows():
        print(".4f")

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str,
                          top_n: int = 15, save_path: str = None):
    """
    Create feature importance visualization.

    Args:
        importance_df: DataFrame with feature importance
        model_name: Name of the model
        top_n: Number of top features to show
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))

    # Plot top N features
    top_features = importance_df.head(top_n)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')

    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_roc_curve(models_dict: Dict[str, Any], X_test: pd.DataFrame,
                  y_test: pd.Series, save_path: str = None):
    """
    Plot ROC curves for multiple models.

    Args:
        models_dict: Dictionary with model names as keys and trained models as values
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))

    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")

    plt.show()


def run_modeling_pipeline(features_dir: str = "data/features",
                         results_dir: str = "results",
                         x_filename: str = None) -> Dict[str, Any]:
    """
    Execute complete modeling pipeline.

    Args:
        features_dir: Directory containing feature files
        results_dir: Directory to save results and plots

    Returns:
        Dictionary with all results and trained models
    """
    print("=== OFFICE APOCALYPSE ALGORITHM - MODELING PIPELINE ===")

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # Load and prepare data
    # Load features. If x_filename is provided, load that file from the features directory
    X, y = load_features(features_dir, x_filename=x_filename)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Analyze feature importance
    lr_importance = analyze_feature_importance(lr_model, X.columns.tolist(), "Logistic Regression")
    rf_importance = analyze_feature_importance(rf_model, X.columns.tolist(), "Random Forest")

    # Create visualizations
    plot_feature_importance(lr_importance, "Logistic Regression",
                          save_path=results_path / "lr_feature_importance.png")
    plot_feature_importance(rf_importance, "Random Forest",
                          save_path=results_path / "rf_feature_importance.png")

    plot_roc_curve({"Logistic Regression": lr_model, "Random Forest": rf_model},
                  X_test, y_test, save_path=results_path / "roc_curves.png")

    # Compile results
    results = {
        'models': {
            'logistic_regression': lr_model,
            'random_forest': rf_model
        },
        'metrics': {
            'logistic_regression': lr_metrics,
            'random_forest': rf_metrics
        },
        'feature_importance': {
            'logistic_regression': lr_importance,
            'random_forest': rf_importance
        },
        'data_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict()
        }
    }

    print("\n=== MODELING PIPELINE COMPLETE ===")
    print(f"Results saved to {results_path}/")

    return results


if __name__ == "__main__":
    try:
        # Run complete modeling pipeline
        results = run_modeling_pipeline()
        print("Modeling pipeline completed successfully!")
    except Exception as e:
        print(f"Error in modeling pipeline: {e}")
        print("Make sure feature files exist in data/features/")