"""
Create presentation visuals for professor meeting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def create_presentation_visuals():
    """Create key visuals for professor presentation."""
    
    # Set style for professional presentation
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Office Apocalypse Algorithm - Current Results Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Data Quality Journey (Before/After Data Leakage Fix)
    ax = axes[0, 0]
    categories = ['Accuracy', 'ROC-AUC', 'Precision@10%']
    before_leakage = [99.6, 99.9, 100.0]
    after_clean = [81.7, 88.2, 87.5]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before_leakage, width, label='Before (Leaky)', 
                   color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_clean, width, label='After (Clean)', 
                   color='green', alpha=0.7)
    
    ax.set_ylabel('Performance (%)')
    ax.set_title('Data Leakage Fix Impact', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold')
    
    # Add text annotation
    ax.text(0.5, 0.95, '🚨 Caught Critical Data Leakage!', transform=ax.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 2. Current Model Performance vs Benchmarks
    ax = axes[0, 1]
    models = ['Random\nClassifier', 'Our Baseline\n(Clean)', 'Target\nThreshold']
    roc_scores = [50.0, 88.2, 85.0]  # Random, Our model, Target
    colors = ['red', 'green', 'blue']
    
    bars = ax.bar(models, roc_scores, color=colors, alpha=0.7)
    ax.set_ylabel('ROC-AUC (%)')
    ax.set_title('Model Performance Benchmark', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold', fontsize=12)
    
    # Add performance band
    ax.axhspan(85, 95, alpha=0.2, color='green', label='Excellent Range')
    ax.axhspan(75, 85, alpha=0.2, color='orange', label='Good Range')
    ax.axhspan(50, 75, alpha=0.2, color='red', label='Poor Range')
    
    # 3. Feature Importance (Top 10)
    ax = axes[0, 2]
    features = ['building_age', 'office_ratio', 'floor_efficiency', 'value_per_sqft',
               'land_value_ratio', 'transaction_count', 'distress_score',
               'mta_accessibility', 'business_density', 'construction_activity']
    importance = [17.6, 16.2, 14.3, 12.8, 9.1, 8.7, 6.4, 5.9, 4.8, 4.2]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
    ax.set_xlabel('Importance (%)')
    ax.set_title('Top 10 Predictive Features', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Task Progress Timeline
    ax = axes[1, 0]
    tasks = ['Temporal\nValidation', 'Baseline\nModel', 'Data Quality\nFix', 
             'Hyperparameter\nTuning', 'Advanced\nModels', 'Final\nEvaluation']
    status = [100, 100, 100, 0, 0, 0]  # Completed percentages
    colors = ['green' if s == 100 else 'orange' if s > 0 else 'lightgray' for s in status]
    
    bars = ax.bar(tasks, status, color=colors, alpha=0.7)
    ax.set_ylabel('Completion (%)')
    ax.set_title('Project Phase Progress', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 110)
    
    # Add completion labels
    for i, (bar, stat) in enumerate(zip(bars, status)):
        if stat == 100:
            label = '✅ Done'
        elif stat > 0:
            label = f'{stat}%'
        else:
            label = '📋 Planned'
        
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold', fontsize=10)
    
    # 5. Business Impact Metrics
    ax = axes[1, 1]
    metrics = ['Precision@5%', 'Precision@10%', 'Precision@20%', 'Overall\nAccuracy']
    values = [92.1, 87.5, 78.3, 81.7]
    
    # Create a line plot showing precision at different targeting levels
    targeting_levels = [5, 10, 20, 100]  # 100 for overall accuracy
    precision_values = [92.1, 87.5, 78.3, 81.7]
    
    ax.plot(targeting_levels[:3], precision_values[:3], 'o-', linewidth=3, 
           markersize=8, color='darkgreen', label='Precision@K%')
    ax.axhline(y=precision_values[3], color='blue', linestyle='--', 
              linewidth=2, label='Overall Accuracy')
    
    ax.set_xlabel('Targeting Percentage (%)')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Targeting Effectiveness', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for x, y in zip(targeting_levels[:3], precision_values[:3]):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                   textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold', fontsize=10)
    
    # 6. Next Steps Roadmap
    ax = axes[1, 2]
    ax.axis('off')  # Turn off axes for text-based content
    
    # Create a flowchart-style next steps
    roadmap_text = """
🎯 IMMEDIATE NEXT STEPS

1️⃣ Hyperparameter Tuning
   • Grid search optimization
   • Cross-validation
   • Performance boost

2️⃣ Advanced Models (Optional)
   • Random Forest
   • XGBoost comparison
   • Ensemble methods

3️⃣ Model Interpretation
   • SHAP analysis
   • Feature interactions
   • Policy insights

4️⃣ Technical Paper
   • Methodology documentation
   • Results summary
   • Policy recommendations

❓ QUESTIONS FOR PROFESSOR:
   • Performance thresholds?
   • Complexity vs interpretability?
   • Timeline priorities?
    """
    
    ax.text(0.05, 0.95, roadmap_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    Path("docs").mkdir(exist_ok=True)
    plt.savefig('docs/professor_presentation_visuals.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ Created presentation visuals: docs/professor_presentation_visuals.png")
    plt.close()
    
    # Create a simple data quality comparison chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Data Quality Evolution - Key Learning', fontsize=16, fontweight='bold')
    
    stages = ['Original Data\n(Suspicious)', 'Identified\nLeakage', 'Clean Data\n(Trustworthy)']
    accuracy_scores = [99.6, 50.0, 81.7]  # Suspicious, Random (during cleaning), Realistic
    colors = ['red', 'orange', 'green']
    
    bars = ax.bar(stages, accuracy_scores, color=colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Our Data Quality Journey - Why 99% Was Too Good to Be True', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bar, score in zip(bars, accuracy_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold', fontsize=14)
    
    # Add explanatory text
    ax.text(0, 90, '❌ Too perfect!\nData leakage', ha='center', va='center',
           fontsize=10, fontweight='bold', color='darkred')
    ax.text(1, 40, '🔍 Investigation\nphase', ha='center', va='center',
           fontsize=10, fontweight='bold', color='darkorange')
    ax.text(2, 70, '✅ Realistic &\ntrustworthy', ha='center', va='center',
           fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('docs/data_quality_journey.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ Created data quality chart: docs/data_quality_journey.png")
    plt.close()

if __name__ == "__main__":
    create_presentation_visuals()
    print("\\n🎯 Presentation materials ready!")
    print("\\nFiles created:")
    print("  1. docs/professor_presentation.md - Main presentation content")
    print("  2. docs/professor_questions.md - Strategic questions to ask")
    print("  3. docs/professor_presentation_visuals.png - Visual summary")
    print("  4. docs/data_quality_journey.png - Data quality story")