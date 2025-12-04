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
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# PACE University Official Colors
PACE_BLUE = '#003C7D'          # Primary PACE Blue
PACE_BLUE_LIGHT = '#0052A5'    # Light PACE Blue
PACE_GOLD = '#FFB81C'          # PACE Gold
PACE_NAVY = '#002855'          # PACE Navy

# PACE Color Palette for Charts
PACE_PALETTE = ['#003C7D', '#0052A5', '#FFB81C', '#002855', '#6699CC']

# Create output directory
output_dir = Path('figures/poster_charts')
output_dir.mkdir(parents=True, exist_ok=True)

def create_chart1_borough_distribution():
    """Chart 1: Office Buildings Distribution by Borough - Pie Chart"""
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx']
    buildings = [2507, 1776, 1619, 705, 584]
    colors = PACE_PALETTE  # Use official PACE colors
    
    fig, ax = plt.subplots(figsize=(16, 12))
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
    plt.savefig(output_dir / 'chart1_borough_distribution.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart1_borough_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 1 saved: Borough Distribution Pie Chart (SVG + PNG)")
    plt.close()

def create_chart2_data_sources():
    """Chart 2: Data Sources Integration Overview - Bar Chart"""
    data_sources = ['NYC PLUTO', 'ACRIS', 'Business\nRegistry', 'MTA\nRidership', 
                    'DOB\nPermits', 'Storefront\nVacancy']
    record_counts = [857736, 1500000, 250000, 3800000, 850000, 12000]
    
    fig, ax = plt.subplots(figsize=(16, 8))
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
    plt.savefig(output_dir / 'chart2_data_sources.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart2_data_sources.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 2 saved: Data Sources Bar Chart (SVG + PNG)")
    plt.close()

def create_chart3_system_architecture():
    """Chart 3: System Architecture Diagram - Flowchart with enhanced clarity"""
    fig, ax = plt.subplots(figsize=(20, 12), dpi=400)
    ax.axis('off')
    
    # Define boxes with PACE colors - gradient from light to dark blue with gold accents
    boxes = [
        # Data Sources
        {'text': '6 Data Sources\n(PLUTO, ACRIS, DOB,\nMTA, Business, Storefront)', 
         'xy': (0.1, 0.7), 'color': '#B3D1E6', 'width': 0.15},  # Light PACE blue
        
        # ETL Pipeline
        {'text': 'ETL Pipeline\n(BBL Standardization,\nTemporal Alignment)', 
         'xy': (0.3, 0.7), 'color': '#6699CC', 'width': 0.15},  # Medium light blue
        
        # Feature Engineering
        {'text': 'Feature Engineering\n(20 Leakage-Free\nFeatures)', 
         'xy': (0.5, 0.7), 'color': '#FFE599', 'width': 0.15},  # PACE gold tint
        
        # XGBoost Model
        {'text': 'XGBoost Model\n(92.41% ROC-AUC)\n7,191 Buildings', 
         'xy': (0.7, 0.7), 'color': '#0052A5', 'width': 0.15},  # PACE blue light
        
        # SHAP Analysis
        {'text': 'SHAP Analysis\n(Feature Importance\n& Explainability)', 
         'xy': (0.5, 0.4), 'color': '#003C7D', 'width': 0.15},  # PACE blue primary
        
        # Dashboard Output
        {'text': 'Streamlit Dashboard\n(Risk Scores,\nInterventions)', 
         'xy': (0.7, 0.4), 'color': '#002855', 'width': 0.15},  # PACE navy
    ]
    
    for box in boxes:
        rect = plt.Rectangle(box['xy'], box['width'], 0.15, 
                           facecolor=box['color'], edgecolor='black', linewidth=4)
        ax.add_patch(rect)
        # Use white text for dark backgrounds, black for light backgrounds
        text_color = 'white' if box['color'] in ['#003C7D', '#002855', '#0052A5'] else 'black'
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + 0.075, 
               box['text'], ha='center', va='center', fontsize=18, weight='bold', 
               fontfamily='sans-serif', color=text_color, antialiased=True)
    
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
                   arrowprops=dict(arrowstyle='->', lw=4, color=PACE_BLUE, 
                                 connectionstyle='arc3,rad=0'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    plt.tight_layout()
    
    # Save as both SVG (vector - never blurry) and PNG (backup)
    plt.savefig(output_dir / 'chart3_system_architecture.svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='svg')
    plt.savefig(output_dir / 'chart3_system_architecture.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Chart 3 saved: System Architecture Diagram (SVG + PNG)")
    plt.close()

def create_chart4_model_comparison():
    """Chart 4: Model Performance Comparison - Horizontal Bar Chart"""
    models = ['XGBoost\n(Champion)', 'Random\nForest', 'Logistic\nRegression']
    roc_auc_scores = [0.9241, 0.9208, 0.8820]
    colors = [PACE_NAVY, PACE_BLUE, PACE_BLUE_LIGHT]  # PACE color gradient
    
    fig, ax = plt.subplots(figsize=(16, 9))
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
    plt.savefig(output_dir / 'chart4_model_comparison.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart4_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 4 saved: Model Performance Comparison (SVG + PNG)")
    plt.close()

def create_chart5_shap_importance():
    """Chart 5: SHAP Feature Importance - Waterfall Plot"""
    features = ['building_age', 'construction_activity', 'officearea', 
                'office_ratio', 'commercial_ratio']
    shap_values = [1.406, 1.149, 0.776, 0.667, 0.568]
    # Use PACE blue gradient for importance
    colors = [PACE_NAVY, PACE_BLUE, PACE_BLUE_LIGHT, '#4A90E2', '#6699CC']
    
    fig, ax = plt.subplots(figsize=(16, 10))
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
    plt.savefig(output_dir / 'chart5_shap_importance.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart5_shap_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 5 saved: SHAP Feature Importance (SVG + PNG)")
    plt.close()

def create_chart6_borough_risk():
    """Chart 6: NYC Borough Risk Heat Map - Color-coded Bar Chart"""
    boroughs = ['Brooklyn', 'Queens', 'Bronx', 'Staten\nIsland', 'Manhattan']
    risk_percentages = [40.9, 32.9, 27.9, 25.5, 22.1]
    building_counts = [1776, 1619, 584, 705, 2507]
    
    # Single color gradient: darker blue = higher risk (academic convention)
    # Create gradient from dark PACE blue to light blue
    from matplotlib.colors import LinearSegmentedColormap
    blues = ['#001F3F', '#003C7D', '#0052A5', '#4A90E2', '#6699CC']
    colors = blues  # Darkest to lightest matches highest to lowest risk
    
    fig, ax = plt.subplots(figsize=(16, 9))
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
    plt.savefig(output_dir / 'chart6_borough_risk.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart6_borough_risk.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 6 saved: Borough Risk Heat Map (SVG + PNG)")
    plt.close()

def create_chart7_business_impact():
    """Chart 7: Business Impact Visualization - Comparison Chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Success Rate Comparison - PACE Colors
    methods = ['Random\nTargeting', 'Model-Driven\nTargeting']
    success_rates = [30, 93]
    colors_success = [PACE_BLUE_LIGHT, PACE_NAVY]  # Light blue for poor, Navy for excellent
    
    bars1 = ax1.bar(methods, success_rates, color=colors_success, edgecolor='black', linewidth=2)
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                f'{rate}%', ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax1.set_ylabel('Success Rate (%)', fontsize=14, weight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_title('Targeting Success Rate', fontsize=14, weight='bold', color=PACE_NAVY)
    ax1.axhline(y=50, color=PACE_GOLD, linestyle='--', linewidth=2, alpha=0.7, label='50% Benchmark')
    ax1.legend(fontsize=11)
    
    # Cost Comparison - PACE Colors
    costs = [5.0, 3.6]
    colors_cost = [PACE_BLUE_LIGHT, PACE_NAVY]  # Light blue for high cost, Navy for low cost
    
    bars2 = ax2.bar(methods, costs, color=colors_cost, edgecolor='black', linewidth=2)
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
                f'${cost}M', ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax2.set_ylabel('Intervention Cost ($M)', fontsize=14, weight='bold')
    ax2.set_ylim(0, 6)
    ax2.set_title('Total Cost for Equivalent Coverage', fontsize=14, weight='bold', color=PACE_NAVY)
    
    # Add improvement annotation with PACE gold
    ax2.text(0.5, 4.5, '$1.4M\nSavings', ha='center', va='center',
            fontsize=14, weight='bold', color=PACE_NAVY,
            bbox=dict(boxstyle='round', facecolor=PACE_GOLD, alpha=0.9, edgecolor=PACE_BLUE, linewidth=3))
    
    fig.suptitle('Business Impact: Model-Driven vs Random Targeting\n3.1× Efficiency Improvement', 
                fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'chart7_business_impact.svg', bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'chart7_business_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 7 saved: Business Impact Comparison (SVG + PNG)")
    plt.close()

def create_chart8_metrics_dashboard():
    """Chart 8: Performance Metrics Dashboard - Professional KPI visualization"""
    fig = plt.figure(figsize=(18, 14), dpi=300, facecolor='white')
    
    # Create grid layout - 4 rows instead of 3
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, 
                         left=0.1, right=0.9, top=0.92, bottom=0.08)
    
    # Title
    fig.text(0.5, 0.96, 'Model Performance Metrics', ha='center', va='top',
            fontsize=32, weight='bold', color=PACE_NAVY, fontfamily='sans-serif')
    
    # ----- CENTER: Main ROC-AUC Metric ----- (ENLARGED for prominence)
    ax_center = fig.add_subplot(gs[1, 1])
    ax_center.axis('off')
    
    # Create circular badge for main metric (LARGER for emphasis)
    circle = plt.Circle((0.5, 0.5), 0.52, color=PACE_BLUE, alpha=0.12, transform=ax_center.transAxes)
    ax_center.add_patch(circle)
    circle_border = plt.Circle((0.5, 0.5), 0.52, fill=False, edgecolor=PACE_BLUE, 
                              linewidth=7, transform=ax_center.transAxes)
    ax_center.add_patch(circle_border)
    
    ax_center.text(0.5, 0.60, '92.41%', ha='center', va='center', 
                  fontsize=64, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                  transform=ax_center.transAxes)
    ax_center.text(0.5, 0.36, 'ROC-AUC', ha='center', va='center', 
                  fontsize=26, weight='bold', color=PACE_BLUE, fontfamily='sans-serif',
                  transform=ax_center.transAxes)
    ax_center.text(0.5, 0.24, 'XGBoost Champion', ha='center', va='center', 
                  fontsize=14, color='#555555', fontfamily='sans-serif', style='italic',
                  transform=ax_center.transAxes)
    
    # ----- TOP ROW: Precision Metrics ----- (ENLARGED)
    precision_metrics = [
        {'value': '93.01%', 'label': 'Precision@10%', 'sublabel': 'Top 10% Targeting', 'pos': gs[0, 0]},
        {'value': '95.12%', 'label': 'Precision@5%', 'sublabel': 'Critical Cases', 'pos': gs[0, 2]},
    ]
    
    for metric in precision_metrics:
        ax = fig.add_subplot(metric['pos'])
        ax.axis('off')
        
        # Background box
        rect = plt.Rectangle((0.05, 0.15), 0.9, 0.7, 
                           facecolor='#F8F9FA', edgecolor=PACE_BLUE_LIGHT, linewidth=4, 
                           transform=ax.transAxes, zorder=1, alpha=0.8)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.65, metric['value'], ha='center', va='center', 
                fontsize=56, weight='bold', color=PACE_BLUE, fontfamily='sans-serif',
                transform=ax.transAxes)
        ax.text(0.5, 0.40, metric['label'], ha='center', va='center', 
                fontsize=18, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                transform=ax.transAxes)
        ax.text(0.5, 0.22, metric['sublabel'], ha='center', va='center', 
                fontsize=13, color='#555555', fontfamily='sans-serif',
                transform=ax.transAxes)
    
    # ----- MIDDLE ROW: Model Info -----
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.axis('off')
    ax_left.text(0.5, 0.75, '7,191', ha='center', va='center', 
                fontsize=42, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                transform=ax_left.transAxes)
    ax_left.text(0.5, 0.48, 'NYC Office Buildings', ha='center', va='center', 
                fontsize=14, weight='bold', color=PACE_BLUE, fontfamily='sans-serif',
                transform=ax_left.transAxes)
    ax_left.text(0.5, 0.32, 'in Dataset', ha='center', va='center', 
                fontsize=12, color='#555555', fontfamily='sans-serif',
                transform=ax_left.transAxes)
    rect = plt.Rectangle((0.1, 0.25), 0.8, 0.6, fill=False, 
                        edgecolor=PACE_GOLD, linewidth=3, transform=ax_left.transAxes)
    ax_left.add_patch(rect)
    
    ax_right = fig.add_subplot(gs[1, 2])
    ax_right.axis('off')
    ax_right.text(0.5, 0.75, '20', ha='center', va='center', 
                 fontsize=42, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                 transform=ax_right.transAxes)
    ax_right.text(0.5, 0.50, 'Engineered\nFeatures', ha='center', va='center', 
                 fontsize=16, weight='bold', color=PACE_BLUE, fontfamily='sans-serif',
                 transform=ax_right.transAxes)
    rect = plt.Rectangle((0.1, 0.25), 0.8, 0.6, fill=False, 
                        edgecolor=PACE_GOLD, linewidth=3, transform=ax_right.transAxes)
    ax_right.add_patch(rect)
    
    # ----- ROW 3: First set of business metrics (2 boxes) -----
    row3_metrics = [
        {'value': '$1.4M', 'label': 'Cost Savings', 'detail': 'Per 1,000 Buildings', 'pos': gs[2, 0], 'color': PACE_GOLD},
        {'value': '68%', 'label': 'Lower Cost/Success', 'detail': '$16.7K → $5.4K per IK', 'pos': gs[2, 2], 'color': PACE_NAVY},
    ]
    
    for metric in row3_metrics:
        ax = fig.add_subplot(metric['pos'])
        ax.axis('off')
        
        # White background with border
        rect = plt.Rectangle((0.05, 0.15), 0.9, 0.7, 
                           facecolor='white', edgecolor='black', linewidth=2,
                           transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.65, metric['value'], ha='center', va='center', 
                fontsize=44, weight='bold', color=metric['color'], fontfamily='sans-serif',
                transform=ax.transAxes)
        ax.text(0.5, 0.42, metric['label'], ha='center', va='center', 
                fontsize=14, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                transform=ax.transAxes)
        ax.text(0.5, 0.25, metric['detail'], ha='center', va='center', 
                fontsize=10, color='#555555', fontfamily='sans-serif',
                transform=ax.transAxes)
    
    # ----- ROW 4: Single centered efficiency metric -----
    row4_metrics = [
        {'value': '2.23×', 'label': 'More Successes', 'detail': 'vs. Random', 'pos': gs[3, 1], 'color': PACE_BLUE}
    ]
    
    for metric in row4_metrics:
        ax = fig.add_subplot(metric['pos'])
        ax.axis('off')
        
        # White background with border
        rect = plt.Rectangle((0.05, 0.15), 0.9, 0.7, 
                           facecolor='white', edgecolor='black', linewidth=2,
                           transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.65, metric['value'], ha='center', va='center', 
                fontsize=44, weight='bold', color=metric['color'], fontfamily='sans-serif',
                transform=ax.transAxes)
        ax.text(0.5, 0.42, metric['label'], ha='center', va='center', 
                fontsize=14, weight='bold', color=PACE_NAVY, fontfamily='sans-serif',
                transform=ax.transAxes)
        if metric['detail']:
            ax.text(0.5, 0.25, metric['detail'], ha='center', va='center', 
                    fontsize=10, color='#555555', fontfamily='sans-serif',
                    transform=ax.transAxes)
    
    plt.savefig(output_dir / 'chart8_metrics_dashboard.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2, format='svg')
    plt.savefig(output_dir / 'chart8_metrics_dashboard.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print("✅ Chart 8 saved: Professional Metrics Dashboard (SVG + PNG)")
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
    create_chart8_metrics_dashboard()
    
    print("\n" + "="*60)
    print("✅ ALL 8 CHARTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCharts saved to: {output_dir.absolute()}")
    print("\nNext Steps:")
    print("1. Review charts in figures/poster_charts/ directory")
    print("2. Insert charts into your poster template")
    print("3. Adjust colors/sizes to match PACE branding if needed")
    print("\n")

if __name__ == "__main__":
    main()
