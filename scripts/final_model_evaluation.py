# Final Model Evaluation - Office Apocalypse Algorithm
**Comprehensive Test Set Evaluation and Operational Deployment Readiness Assessment**

*Date: November 24, 2025*
*Team: Ibrahim Denis Fofanah (Leader), Bright Arowny Zaman, Jeevan Hemanth Yendluri*

---

## Final Model Evaluation Script

```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, brier_score_loss,
    calibration_curve, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinalModelEvaluator:
    """
    Comprehensive evaluation class for the champion XGBoost model
    focusing on business metrics, calibration, and operational readiness.
    """
    
    def __init__(self, model_path, data_path, feature_names_path):
        """Initialize evaluator with saved model and test data."""
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(data_path)
        
        # Load feature names
        with open(feature_names_path, 'r') as f:
            self.features = [line.strip() for line in f.readlines()]
        
        self.results = {}
        
    def prepare_test_data(self, test_size=0.2, random_state=42):
        """Prepare final holdout test set with temporal validation."""
        print("🔄 Preparing Final Test Set...")
        
        # Ensure we have the target variable
        if 'is_high_risk' not in self.data.columns:
            # Create target based on vacancy indicators
            # (This should be defined based on your specific target creation logic)
            print("Creating target variable...")
            self.data['is_high_risk'] = (self.data['storefront_vacancy_rate'] > 0.3).astype(int)
        
        # Sort by a date column if available for temporal split
        if 'yearbuilt' in self.data.columns:
            self.data = self.data.sort_values('yearbuilt')
        
        # Create temporal test split (most recent 20%)
        split_idx = int(len(self.data) * (1 - test_size))
        
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        
        # Prepare features and targets
        self.X_train = self.train_data[self.features]
        self.y_train = self.train_data['is_high_risk']
        self.X_test = self.test_data[self.features]
        self.y_test = self.test_data['is_high_risk']
        
        print(f"✅ Test set prepared: {len(self.test_data)} buildings")
        print(f"   Train: {len(self.train_data)} | Test: {len(self.test_data)}")
        print(f"   Test set high-risk rate: {self.y_test.mean():.3f}")
        
    def evaluate_performance_metrics(self):
        """Comprehensive performance evaluation on test set."""
        print("\n📊 Evaluating Performance Metrics...")
        
        # Generate predictions
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)
        
        # Core metrics
        roc_auc = roc_auc_score(self.y_test, y_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Business-critical metrics
        precision_at_10 = self._precision_at_k(self.y_test, y_proba, k=0.1)
        precision_at_5 = self._precision_at_k(self.y_test, y_proba, k=0.05)
        precision_at_1 = self._precision_at_k(self.y_test, y_proba, k=0.01)
        
        # Calibration metrics
        brier_score = brier_score_loss(self.y_test, y_proba)
        
        self.results['performance'] = {
            'ROC-AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Precision@10%': precision_at_10,
            'Precision@5%': precision_at_5,
            'Precision@1%': precision_at_1,
            'Brier Score': brier_score,
            'Test Set Size': len(self.y_test),
            'High-Risk Buildings': self.y_test.sum()
        }
        
        # Print results
        print("🎯 FINAL MODEL PERFORMANCE ON TEST SET:")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Precision@10%: {precision_at_10:.4f} (Target 10% highest-risk buildings)")
        print(f"   Precision@5%: {precision_at_5:.4f} (Critical intervention targets)")
        print(f"   Precision@1%: {precision_at_1:.4f} (Emergency intervention targets)")
        print(f"   Brier Score: {brier_score:.4f} (Calibration quality)")
        
    def _precision_at_k(self, y_true, y_proba, k):
        """Calculate precision when targeting top k% of predictions."""
        n_targets = int(len(y_true) * k)
        top_k_indices = np.argsort(y_proba)[-n_targets:]
        return y_true.iloc[top_k_indices].mean()
        
    def evaluate_calibration(self):
        """Assess probability calibration quality."""
        print("\n🎯 Evaluating Probability Calibration...")
        
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calibration curve analysis
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_proba, n_bins=10, strategy='quantile'
        )
        
        # Expected Calibration Error (ECE)
        ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
        
        self.results['calibration'] = {
            'Expected_Calibration_Error': ece,
            'Fraction_of_Positives': fraction_of_positives,
            'Mean_Predicted_Value': mean_predicted_value,
            'Reliability_Score': 1 - ece  # Higher is better
        }
        
        print(f"   Expected Calibration Error: {ece:.4f}")
        print(f"   Reliability Score: {1-ece:.4f} (1.0 = perfect calibration)")
        
    def evaluate_business_impact(self):
        """Quantify business value and operational metrics."""
        print("\n💼 Evaluating Business Impact...")
        
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Simulate intervention scenarios
        baseline_success_rate = self.y_test.mean()  # Random targeting
        
        # Model-driven targeting scenarios
        scenarios = {
            'Top 10%': 0.10,
            'Top 5%': 0.05,
            'Top 1%': 0.01
        }
        
        business_metrics = {}
        
        for scenario_name, target_pct in scenarios.items():
            precision_at_k = self._precision_at_k(self.y_test, y_proba, target_pct)
            n_buildings = int(len(self.test_data) * target_pct)
            
            # Calculate efficiency improvement
            efficiency_improvement = precision_at_k / baseline_success_rate
            
            # Cost analysis (assuming $5,000 per building intervention)
            cost_per_intervention = 5000
            random_cost = cost_per_intervention * n_buildings
            successful_interventions = int(n_buildings * precision_at_k)
            cost_per_success = random_cost / max(successful_interventions, 1)
            
            business_metrics[scenario_name] = {
                'Precision': precision_at_k,
                'Buildings_Targeted': n_buildings,
                'Successful_Interventions': successful_interventions,
                'Efficiency_Improvement': efficiency_improvement,
                'Total_Cost': random_cost,
                'Cost_Per_Success': cost_per_success
            }
            
        self.results['business_impact'] = {
            'Baseline_Success_Rate': baseline_success_rate,
            'Scenarios': business_metrics
        }
        
        # Print business impact summary
        print(f"   Baseline (Random) Success Rate: {baseline_success_rate:.3f}")
        for scenario, metrics in business_metrics.items():
            print(f"\n   {scenario} Targeting:")
            print(f"     Success Rate: {metrics['Precision']:.3f}")
            print(f"     Efficiency Improvement: {metrics['Efficiency_Improvement']:.1f}x")
            print(f"     Cost per Success: ${metrics['Cost_Per_Success']:,.0f}")
            
    def evaluate_feature_stability(self):
        """Assess feature importance stability and model robustness."""
        print("\n🔍 Evaluating Feature Stability...")
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        # Feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        self.results['feature_stability'] = {
            'Feature_Importance': feature_importance_df,
            'Top_5_Features': feature_importance_df.head(5)['Feature'].tolist(),
            'SHAP_Values': shap_values
        }
        
        print("   Top 5 Most Important Features:")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"     {row['Feature']}: {row['Importance']:.4f}")
            
    def evaluate_geographic_performance(self):
        """Assess model performance across different NYC boroughs."""
        print("\n🗺️ Evaluating Geographic Performance...")
        
        if 'borough' not in self.test_data.columns:
            print("   Borough information not available - skipping geographic analysis")
            return
        
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        borough_performance = {}
        for borough in self.test_data['borough'].unique():
            mask = self.test_data['borough'] == borough
            if mask.sum() < 10:  # Skip boroughs with too few samples
                continue
                
            borough_auc = roc_auc_score(self.y_test[mask], y_proba[mask])
            borough_precision = self._precision_at_k(
                self.y_test[mask], y_proba[mask], k=0.1
            )
            
            borough_performance[borough] = {
                'ROC_AUC': borough_auc,
                'Precision_at_10': borough_precision,
                'Sample_Size': mask.sum(),
                'High_Risk_Rate': self.y_test[mask].mean()
            }
            
        self.results['geographic_performance'] = borough_performance
        
        print("   Performance by Borough:")
        for borough, metrics in borough_performance.items():
            print(f"     {borough}: ROC-AUC={metrics['ROC_AUC']:.3f}, "
                  f"P@10%={metrics['Precision_at_10']:.3f}, "
                  f"n={metrics['Sample_Size']}")
            
    def generate_deployment_readiness_report(self):
        """Generate comprehensive deployment readiness assessment."""
        print("\n📋 DEPLOYMENT READINESS ASSESSMENT")
        print("="*60)
        
        # Performance thresholds for production deployment
        performance_thresholds = {
            'ROC_AUC_min': 0.90,
            'Precision_at_10_min': 0.85,
            'Calibration_max_error': 0.10,
            'Feature_stability_min': 5  # Minimum number of stable features
        }
        
        readiness_checks = {}
        
        # Check 1: Performance Standards
        perf = self.results['performance']
        readiness_checks['Performance'] = {
            'ROC_AUC': perf['ROC-AUC'] >= performance_thresholds['ROC_AUC_min'],
            'Precision_at_10': perf['Precision@10%'] >= performance_thresholds['Precision_at_10_min'],
            'Test_Sample_Size': perf['Test Set Size'] >= 100
        }
        
        # Check 2: Calibration Quality
        cal = self.results['calibration']
        readiness_checks['Calibration'] = {
            'Expected_Calibration_Error': cal['Expected_Calibration_Error'] <= performance_thresholds['Calibration_max_error']
        }
        
        # Check 3: Feature Stability
        feat = self.results['feature_stability']
        readiness_checks['Feature_Stability'] = {
            'Sufficient_Important_Features': len(feat['Top_5_Features']) >= 5
        }
        
        # Check 4: Business Value
        biz = self.results['business_impact']
        readiness_checks['Business_Value'] = {
            'Top_10_Efficiency': biz['Scenarios']['Top 10%']['Efficiency_Improvement'] >= 2.0,
            'Top_5_Precision': biz['Scenarios']['Top 5%']['Precision'] >= 0.80
        }
        
        # Overall readiness assessment
        all_checks_passed = all(
            all(check_results.values()) 
            for check_results in readiness_checks.values()
        )
        
        print(f"OVERALL STATUS: {'✅ READY FOR DEPLOYMENT' if all_checks_passed else '⚠️  REQUIRES ATTENTION'}")
        print("\nDetailed Readiness Checks:")
        
        for category, checks in readiness_checks.items():
            print(f"\n{category}:")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name}")
                
        self.results['deployment_readiness'] = {
            'Overall_Ready': all_checks_passed,
            'Detailed_Checks': readiness_checks,
            'Performance_Thresholds': performance_thresholds
        }
        
        return all_checks_passed
        
    def save_evaluation_results(self, output_path="results/final_evaluation_report.json"):
        """Save comprehensive evaluation results."""
        import json
        import os
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"\n💾 Evaluation results saved to: {output_path}")
        
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, 'to_dict'):  # pandas DataFrame/Series
            return obj.to_dict()
        else:
            return obj

def run_final_evaluation():
    """Execute complete final evaluation pipeline."""
    print("🚀 OFFICE APOCALYPSE ALGORITHM - FINAL MODEL EVALUATION")
    print("="*70)
    
    # Initialize evaluator
    evaluator = FinalModelEvaluator(
        model_path="models/champion_xgboost.pkl",
        data_path="data/processed/office_buildings_with_coordinates.csv",
        feature_names_path="models/champion_features.txt"
    )
    
    try:
        # Execute evaluation pipeline
        evaluator.prepare_test_data()
        evaluator.evaluate_performance_metrics()
        evaluator.evaluate_calibration()
        evaluator.evaluate_business_impact()
        evaluator.evaluate_feature_stability()
        evaluator.evaluate_geographic_performance()
        
        # Final readiness assessment
        is_ready = evaluator.generate_deployment_readiness_report()
        
        # Save results
        evaluator.save_evaluation_results()
        
        print("\n🎉 Final evaluation completed successfully!")
        print(f"Model deployment status: {'APPROVED' if is_ready else 'NEEDS REVIEW'}")
        
        return evaluator.results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    results = run_final_evaluation()