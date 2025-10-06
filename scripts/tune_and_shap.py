"""Hyperparameter tuning and SHAP explainability script

Usage: run with the project venv python. Writes outputs to results/
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print('Loading features...')
    X_path = Path('data/features/X_features_with_storefronts.csv')
    y_path = Path('data/features/y_target.csv')
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError('Feature files not found; run feature engineering first')

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]

    print(f'Loaded X: {X.shape}, y: {y.shape}')

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Create tuning sample (limit size for speed)
    tune_n = 100000
    if len(X_train) > tune_n:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=tune_n, stratify=y_train, random_state=42
        )
    else:
        X_tune, y_tune = X_train, y_train

    print(f'Tuning on {len(X_tune)} samples (train full size {len(X_train)})')

    # Random Forest with randomized search
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [6, 8, 10, 12, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.2, 0.5]
    }

    base_rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print('Starting RandomizedSearchCV...')
    rs.fit(X_tune, y_tune)
    print('RandomizedSearchCV complete')

    best = rs.best_estimator_
    best_params = rs.best_params_
    print('Best params:', best_params)

    # Retrain best on full train set
    print('Retraining best estimator on full train set...')
    best.fit(X_train, y_train)

    # Evaluate on test
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_proba))
    }

    print('Test metrics:', metrics)

    # Save model and results
    joblib.dump(best, results_dir / 'rf_tuned.joblib')
    with open(results_dir / 'rf_tuning_results.json', 'w') as f:
        json.dump({'best_params': best_params, 'metrics': metrics}, f, indent=2)

    # SHAP explainability (sample for speed)
    try:
        import shap

        sample_n = min(5000, len(X_test))
        sample_idx = X_test.sample(n=sample_n, random_state=42).index
        X_shap = X_test.loc[sample_idx]

        print('Computing SHAP values (this may take a while)...')
        explainer = shap.TreeExplainer(best)
        shap_values = explainer.shap_values(X_shap)

        # shap_values may be list [class0, class1] for classification
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_shap, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_summary_bar.png', dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, show=False)
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_summary_dot.png', dpi=200, bbox_inches='tight')
        plt.close()

        print('SHAP plots saved')
    except Exception as e:
        print('SHAP failed or not installed:', e)

    print('Tuning + SHAP complete. Artifacts saved to results/')


if __name__ == '__main__':
    main()
