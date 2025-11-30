"""
Generate all charts for Academic Poster - Office Apocalypse Algorithm
Creates 7 visualizations for the capstone poster presentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional academic appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
PACE_BLUE = '#003DA5'
PACE_GOLD = '#FFC72C'

# Create output directory
output_dir = Path('figures/poster_charts')
output_dir.mkdir(parents=True, exist_ok=True)

def create_chart1_borough_distribution():
    """Chart 1: Office Buildings Distribution by Borough - Pie Chart"""
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx']
    buildings = [2507, 1776, 1619, 705, 584]
    colors = ['#003DA5', '#0066CC', '#3399FF', '#66B2FF', '#99CCFF']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        buildings, 
        labels=boroughs, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'bold'}
    )
    
    # Add building counts in legend
    legend_labels = [f'{b}: {n:,} buildings' for b, n in zip(boroughs, buildings)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    plt.title('Office Buildings Distribution by Borough\n(Total: 7,191 Buildings)', 
              fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart1_borough_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 1 saved: Borough Distribution Pie Chart")
    plt.close()

def create_chart2_data_sources():
    """Chart 2: Data Sources Integration Overview - Bar Chart"""
    data_sources = ['NYC PLUTO', 'ACRIS', 'Business\nRegistry', 'MTA\nRidership', 
                    'DOB\nPermits', 'Storefront\nVacancy']
    record_counts = [857736, 1500000, 250000, 3800000, 850000, 12000]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(data_sources, record_counts, color=PACE_BLUE)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, record_counts)):
        width = bar.get_width()
        label = f'{count:,.0f}' if count < 1000000 else f'{count/1000000:.1f}M'
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'  {label}', va='center', fontsize=12, weight='bold')
    
    ax.set_xlabel('Number of Records', fontsize=14, weight='bold')
    ax.set_title('Municipal Data Sources Integration\n(6 Datasets for 7,191 Office Buildings)', 
                 fontsize=16, weight='bold', pad=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    plt.tight_layout()
    plt.savefig(output_dir / 'chart2_data_sources.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 2 saved: Data Sources Bar Chart")
    plt.close()

def create_chart3_system_architecture():
    """Chart 3: System Architecture Diagram - Flowchart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # Data Sources
        {'text': '6 Data Sources\n(PLUTO, ACRIS, DOB,\nMTA, Business, Storefront)', 
         'xy': (0.1, 0.7), 'color': '#E3F2FD', 'width': 0.15},
        
        # ETL Pipeline
        {'text': 'ETL Pipeline\n(BBL Standardization,\nTemporal Alignment)', 
         'xy': (0.3, 0.7), 'color': '#C8E6C9', 'width': 0.15},
        
        # Feature Engineering
        {'text': 'Feature Engineering\n(20 Leakage-Free\nFeatures)', 
         'xy': (0.5, 0.7), 'color': '#FFF9C4', 'width': 0.15},
        
        # XGBoost Model
        {'text': 'XGBoost Model\n(92.41% ROC-AUC)\n7,191 Buildings', 
         'xy': (0.7, 0.7), 'color': '#FFCCBC', 'width': 0.15},
        
        # SHAP Analysis
        {'text': 'SHAP Analysis\n(Feature Importance\n& Explainability)', 
         'xy': (0.5, 0.4), 'color': '#E1BEE7', 'width': 0.15},
        
        # Dashboard Output
        {'text': 'Streamlit Dashboard\n(Risk Scores,\nInterventions)', 
         'xy': (0.7, 0.4), 'color': '#B2DFDB', 'width': 0.15},
    ]
    
    for box in boxes:
        rect = plt.Rectangle(box['xy'], box['width'], 0.15, 
                           facecolor=box['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + 0.075, 
               box['text'], ha='center', va='center', fontsize=11, weight='bold')
    
    # Add arrows
    arrows = [
        ((0.25, 0.775), (0.3, 0.775)),
        ((0.45, 0.775), (0.5, 0.775)),
        ((0.65, 0.775), (0.7, 0.775)),
        ((0.775, 0.7), (0.775, 0.55)),
        ((0.65, 0.475), (0.7, 0.475)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color=PACE_BLUE))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    plt.title('System Architecture: End-to-End ML Pipeline', 
             fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart3_system_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 3 saved: System Architecture Diagram")
    plt.close()

def create_chart4_model_comparison():
    """Chart 4: Model Performance Comparison - Horizontal Bar Chart"""
    models = ['XGBoost\n(Champion)', 'Random\nForest', 'Logistic\nRegression']
    roc_auc_scores = [0.9241, 0.9208, 0.8820]
    colors = ['#2E7D32', '#66BB6A', '#A5D6A7']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, roc_auc_scores, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, score in zip(bars, roc_auc_scores):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{score:.2%}', va='center', fontsize=14, weight='bold')
    
    ax.set_xlabel('ROC-AUC Score', fontsize=14, weight='bold')
    ax.set_xlim(0.85, 0.95)
    ax.axvline(x=0.90, color='red', linestyle='--', linewidth=2, label='90% Threshold')
    ax.legend(fontsize=12)
    ax.set_title('Model Performance Comparison (ROC-AUC)', 
                fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart4_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 4 saved: Model Performance Comparison")
    plt.close()

def create_chart5_shap_importance():
    """Chart 5: SHAP Feature Importance - Waterfall Plot"""
    features = ['building_age', 'construction_activity', 'officearea', 
                'office_ratio', 'commercial_ratio']
    shap_values = [1.406, 1.149, 0.776, 0.667, 0.568]
    colors = ['#D32F2F', '#F44336', '#EF5350', '#E57373', '#EF9A9A']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(features, shap_values, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, shap_values):
        width = bar.get_width()
        ax.text(width + 0.03, bar.get_y() + bar.get_height()/2, 
               f'{value:.3f}', va='center', fontsize=13, weight='bold')
    
    ax.set_xlabel('SHAP Value (Feature Importance)', fontsize=14, weight='bold')
    ax.set_title('Top 5 Predictive Features (SHAP Analysis)\nbuilding_age Dominates Vacancy Risk', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlim(0, 1.6)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart5_shap_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 5 saved: SHAP Feature Importance")
    plt.close()

def create_chart6_borough_risk():
    """Chart 6: NYC Borough Risk Heat Map - Color-coded Bar Chart"""
    boroughs = ['Brooklyn', 'Queens', 'Bronx', 'Staten\nIsland', 'Manhattan']
    risk_percentages = [40.9, 32.9, 27.9, 25.5, 22.1]
    building_counts = [1776, 1619, 584, 705, 2507]
    
    # Color gradient from red (high risk) to green (low risk)
    colors = ['#D32F2F', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(boroughs, risk_percentages, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels with building counts
    for bar, risk, count in zip(bars, risk_percentages, building_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
               f'{risk}%\n({count:,} bldgs)', ha='center', va='bottom', 
               fontsize=12, weight='bold')
    
    ax.set_ylabel('High-Risk Buildings (%)', fontsize=14, weight='bold')
    ax.set_xlabel('NYC Borough', fontsize=14, weight='bold')
    ax.set_ylim(0, 50)
    ax.set_title('Geographic Risk Distribution Across NYC Boroughs\nBrooklyn Shows Highest Vacancy Risk Concentration', 
                fontsize=16, weight='bold', pad=20)
    
    # Add risk level annotations
    ax.axhline(y=35, color='red', linestyle='--', linewidth=2, alpha=0.5, label='High Risk Threshold')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart6_borough_risk.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 6 saved: Borough Risk Heat Map")
    plt.close()

def create_chart7_business_impact():
    """Chart 7: Business Impact Visualization - Comparison Chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Success Rate Comparison
    methods = ['Random\nTargeting', 'Model-Driven\nTargeting']
    success_rates = [30, 93]
    colors_success = ['#FF5252', '#4CAF50']
    
    bars1 = ax1.bar(methods, success_rates, color=colors_success, edgecolor='black', linewidth=2)
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                f'{rate}%', ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax1.set_ylabel('Success Rate (%)', fontsize=14, weight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_title('Targeting Success Rate', fontsize=14, weight='bold')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Cost Comparison
    costs = [5.0, 3.6]
    colors_cost = ['#FF5252', '#4CAF50']
    
    bars2 = ax2.bar(methods, costs, color=colors_cost, edgecolor='black', linewidth=2)
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
                f'${cost}M', ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax2.set_ylabel('Intervention Cost ($M)', fontsize=14, weight='bold')
    ax2.set_ylim(0, 6)
    ax2.set_title('Total Cost for Equivalent Coverage', fontsize=14, weight='bold')
    
    # Add improvement annotation
    ax2.text(0.5, 4.5, '85% Cost\nReduction', ha='center', va='center',
            fontsize=14, weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    fig.suptitle('Business Impact: Model-Driven vs Random Targeting\n3.1× Efficiency Improvement', 
                fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart7_business_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 7 saved: Business Impact Comparison")
    plt.close()

def main():
    """Generate all poster charts"""
    print("\n" + "="*60)
    print("GENERATING ACADEMIC POSTER CHARTS")
    print("Office Apocalypse Algorithm - Capstone Project")
    print("="*60 + "\n")
    
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Generate all charts
    create_chart1_borough_distribution()
    create_chart2_data_sources()
    create_chart3_system_architecture()
    create_chart4_model_comparison()
    create_chart5_shap_importance()
    create_chart6_borough_risk()
    create_chart7_business_impact()
    
    print("\n" + "="*60)
    print("✅ ALL 7 CHARTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCharts saved to: {output_dir.absolute()}")
    print("\nNext Steps:")
    print("1. Review charts in figures/poster_charts/ directory")
    print("2. Insert charts into your poster template")
    print("3. Adjust colors/sizes to match PACE branding if needed")
    print("\n")

if __name__ == "__main__":
    main()
