import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the champion model and test data
champion_model = joblib.load('models/champion_model.joblib')
X_test = pd.read_csv('models/X_test.csv', index_col=0)

# Get feature importance (coefficients for logistic regression)
feature_names = X_test.columns
feature_importance = np.abs(champion_model.coef_[0])

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Create feature importance visualization
plt.figure(figsize=(12, 10))

# Top 15 features
top_features = importance_df.head(15)

# Create horizontal bar plot
bars = plt.barh(range(len(top_features)), top_features['importance'], 
                color='steelblue', alpha=0.8)

# Customize the plot
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance (|Coefficient|)', fontsize=12, fontweight='bold')
plt.title('Top 15 Feature Importance - Champion Model\nOffice Building Vacancy Risk Prediction', 
          fontsize=14, fontweight='bold', pad=20)

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/feature_analysis/champion_model_feature_importance.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print('✅ Champion model feature importance saved to results/feature_analysis/')
print('📊 Top 5 Most Important Features:')
for i, (idx, row) in enumerate(top_features.head(5).iterrows(), 1):
    print(f'   {i}. {row["feature"]}: {row["importance"]:.4f}')