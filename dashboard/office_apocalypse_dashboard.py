#!/usr/bin/env python3
"""
Office Apocalypse Algorithm - Interactive Dashboard
Streamlit web application for vacancy risk prediction

Champion Model: XGBoost (92.41% ROC-AUC)
Version: 2.0 (Simplified feature display - SHAP removed for stability)
Build: 20251201-stable
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Get the absolute path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from dashboard to project root

# Page configuration
st.set_page_config(
    page_title="Office Apocalypse Algorithm | PACE University",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Office Apocalypse Algorithm - NYC Building Vacancy Risk Prediction\n\nTeam: Ibrahim Denis Fofanah, Bright Arowny Zaman, Jeevan Hemanth Yendluri\nAdvisor: Dr. Krishna Bathula"
    }
)

# Custom CSS - PACE University Blue Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@700;800&display=swap');
    
    /* PACE University Official Colors */
    :root {
        --pace-blue: #003C7D;
        --pace-blue-light: #0052A5;
        --pace-gold: #FFB81C;
        --pace-navy: #002855;
    }
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Main Header with PACE Blue */
    .main-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #003C7D 0%, #0052A5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #002855;
        margin-bottom: 2rem;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Champion Badge - Gold PACE Colors */
    .champion-badge {
        background: linear-gradient(135deg, #FFB81C 0%, #FFA500 100%);
        color: #003C7D;
        padding: 0.8rem 2rem;
        border-radius: 2rem;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(255, 184, 28, 0.5);
        border: 3px solid #003C7D;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        animation: pulse 2.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 20px rgba(255, 184, 28, 0.5); }
        50% { transform: scale(1.05); box-shadow: 0 12px 30px rgba(255, 184, 28, 0.7); }
    }
    
    /* Team Badge - PACE Blue */
    .team-badge {
        background: linear-gradient(135deg, #003C7D 0%, #0052A5 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 10px rgba(0, 60, 125, 0.3);
        transition: all 0.3s ease;
    }
    
    .team-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0, 60, 125, 0.5);
    }
    
    /* Enhanced Metric Containers */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #003C7D;
        box-shadow: 0 4px 15px rgba(0, 60, 125, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 60, 125, 0.25);
    }
    
    /* Risk Level Styling */
    .high-risk {
        color: #dc3545;
        font-weight: 700;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 2px 4px rgba(220, 53, 69, 0.3);
    }
    
    .medium-risk {
        color: #FFB81C;
        font-weight: 700;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 2px 4px rgba(255, 184, 28, 0.3);
    }
    
    .low-risk {
        color: #28a745;
        font-weight: 700;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
    }
    
    /* Sidebar Styling - PACE Blue Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003C7D 0%, #002855 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    
    /* Info Boxes */
    .stAlert {
        border-radius: 1rem;
        border-left: 5px solid #003C7D;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white !important;
        color: #2c3e50 !important;
    }
    
    .stAlert p, .stAlert li, .stAlert span {
        color: #2c3e50 !important;
        font-size: 1rem !important;
    }
    
    /* Buttons - PACE Blue */
    .stButton > button {
        background: linear-gradient(135deg, #003C7D 0%, #0052A5 100%);
        color: white;
        border-radius: 0.8rem;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        border: 2px solid #FFB81C;
        box-shadow: 0 4px 15px rgba(0, 60, 125, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 60, 125, 0.6);
        border-color: #003C7D;
    }
    
    /* Download Button - Gold Accent */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #FFB81C 0%, #FFA500 100%);
        color: #003C7D;
        border-radius: 0.8rem;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        border: 2px solid #003C7D;
        box-shadow: 0 4px 15px rgba(255, 184, 28, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 184, 28, 0.6);
    }
    
    /* Section Headers - PACE Blue */
    h2, h3 {
        color: #003C7D;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 0.8rem;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #003C7D;
        font-size: 0.95rem !important;
    }
    
    .dataframe thead tr th {
        background-color: #003C7D !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.8rem !important;
    }
    
    .dataframe tbody tr td {
        color: #2c3e50 !important;
        padding: 0.6rem !important;
        font-size: 0.95rem !important;
    }
    
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f8f9fa !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e3f2fd !important;
    }
    
    /* Metrics - PACE Blue */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #003C7D;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Selectbox and Slider */
    .stSelectbox, .stSlider {
        border-radius: 0.8rem;
    }
    
    /* Footer - PACE Branded */
    .footer {
        text-align: center;
        padding: 2.5rem;
        margin-top: 3rem;
        border-top: 3px solid #003C7D;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 1rem;
    }
    
    .footer h3 {
        color: #003C7D;
        margin-bottom: 1rem;
    }
    
    .footer a {
        color: #003C7D;
        font-weight: 600;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #FFB81C;
    }
    
    /* Spinner - PACE Blue */
    .stSpinner > div {
        border-top-color: #003C7D !important;
    }
    
    /* Progress Bar - PACE Colors */
    .stProgress > div > div {
        background-color: #003C7D;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background-color: rgba(0, 60, 125, 0.1);
        border-left-color: #003C7D;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed dataset and predictions."""
    try:
        # Try to load dataset with coordinates first
        data_path_coords = os.path.join(project_root, 'data', 'processed', 'office_buildings_with_coordinates.csv')
        if os.path.exists(data_path_coords):
            df = pd.read_csv(data_path_coords)
            st.success("✅ Loaded dataset with geographic coordinates!")
        else:
            # Fallback to original dataset
            data_path = os.path.join(project_root, 'data', 'processed', 'office_buildings_clean.csv')
            df = pd.read_csv(data_path)
            st.info("ℹ️ Loaded dataset without coordinates. Geographic mapping not available.")
        
        # Clean any string-formatted numeric columns (fix for SHAP visualization)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert string columns to numeric
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Create target if not exists
        if 'target_high_vacancy_risk' not in df.columns:
            if 'vacancy_risk_alert' in df.columns:
                df['target_high_vacancy_risk'] = (
                    df['vacancy_risk_alert'].isin(['Orange', 'Red'])
                ).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_champion_model():
    """Load the champion XGBoost model."""
    try:
        # Load champion model using absolute path
        model_path = os.path.join(project_root, 'models', 'champion_xgboost.pkl')
        model = joblib.load(model_path)
        
        # Load feature names using absolute path
        features_path = os.path.join(project_root, 'models', 'champion_features.txt')
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        return model, features
    except Exception as e:
        st.error(f"Error loading champion model: {e}")
        return None, None

def get_risk_category(probability):
    """Convert probability to risk category."""
    if probability >= 0.7:
        return "🔴 HIGH", "high-risk"
    elif probability >= 0.4:
        return "🟡 MEDIUM", "medium-risk"
    else:
        return "🟢 LOW", "low-risk"

def create_risk_gauge(probability):
    """Create a professional risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "Vacancy Risk Score",
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Inter'}
        },
        delta = {'reference': 50, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
        number = {'suffix': "%", 'font': {'size': 48, 'color': '#003C7D', 'family': 'Montserrat'}},
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickwidth': 2,
                'tickcolor': "#003C7D"
            },
            'bar': {'color': "#003C7D", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e9ecef",
            'steps': [
                {'range': [0, 40], 'color': "#d4edda"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 4},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    fig.update_layout(
        height=350,
        font={'family': 'Inter'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_feature_importance_plot(shap_values, feature_names, building_data):
    """Create SHAP feature importance plot for a specific building."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get SHAP values for this building
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Clean building_data to ensure all values are floats
    clean_building_data = []
    for val in building_data:
        try:
            if isinstance(val, (list, np.ndarray)):
                val = float(val[0]) if len(val) > 0 else 0.0
            elif isinstance(val, str):
                # Remove brackets and convert
                val = float(val.strip('[]'))
            else:
                val = float(val)
        except (ValueError, TypeError, AttributeError):
            val = 0.0
        clean_building_data.append(val)
    
    # Sort features by absolute SHAP value
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'feature_value': clean_building_data
    })
    importance_df['abs_shap'] = np.abs(importance_df['shap_value'])
    importance_df = importance_df.sort_values('abs_shap', ascending=True).tail(10)
    
    # Create horizontal bar plot
    colors = ['red' if x > 0 else 'blue' for x in importance_df['shap_value']]
    bars = ax.barh(range(len(importance_df)), importance_df['shap_value'], color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in importance_df['feature']])
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title('Top 10 Features Driving This Prediction', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        feature_val = importance_df.iloc[i]['feature_value']
        
        # Handle various data types safely
        try:
            if isinstance(feature_val, (list, np.ndarray)):
                feature_val = float(feature_val[0]) if len(feature_val) > 0 else 0.0
            else:
                feature_val = float(feature_val)
            
            # Format based on magnitude
            if abs(feature_val) >= 1000:
                val_text = f'{feature_val:,.0f}'
            elif abs(feature_val) >= 1:
                val_text = f'{feature_val:.2f}'
            else:
                val_text = f'{feature_val:.4f}'
        except (ValueError, TypeError):
            val_text = str(feature_val)
        
        ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                val_text, ha='left' if width > 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_risk_distribution_plot(df, predictions):
    """Create professional risk distribution visualization."""
    risk_df = pd.DataFrame({
        'Probability': predictions,
        'Borough': df['borough'] if 'borough' in df.columns else 'Unknown'
    })
    
    # PACE University color palette
    colors = {
        'Manhattan': '#003C7D',
        'Brooklyn': '#0052A5',
        'Queens': '#FFB81C',
        'Bronx': '#002855',
        'Staten Island': '#6699CC'
    }
    
    fig = px.histogram(
        risk_df, 
        x='Probability', 
        color='Borough',
        title='<b>Vacancy Risk Distribution Across NYC Boroughs</b>',
        nbins=40,
        labels={'Probability': 'Vacancy Risk Probability', 'count': 'Number of Buildings'},
        color_discrete_map=colors,
        barmode='stack'
    )
    
    # Add risk threshold lines with annotations
    fig.add_vline(
        x=0.4, 
        line_dash="dash", 
        line_color="#ffc107", 
        line_width=3,
        annotation_text="Medium Risk", 
        annotation_position="top",
        annotation_font_color="#ffc107",
        annotation_font_size=14
    )
    fig.add_vline(
        x=0.7, 
        line_dash="dash", 
        line_color="#dc3545", 
        line_width=3,
        annotation_text="High Risk", 
        annotation_position="top",
        annotation_font_color="#dc3545",
        annotation_font_size=14
    )
    
    fig.update_layout(
        height=450,
        font={'family': 'Inter', 'size': 12},
        title_font_size=20,
        title_font_color='#2c3e50',
        paper_bgcolor='rgba(248, 249, 250, 0.5)',
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_geographic_risk_map(df, predictions):
    """Create geographic risk visualization."""
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Filter out invalid coordinates
        valid_coords = (
            (df['latitude'].notna()) & 
            (df['longitude'].notna()) & 
            (df['latitude'] != 0) & 
            (df['longitude'] != 0)
        )
        
        if valid_coords.sum() > 0:
            map_df = df[valid_coords].copy()
            map_df['risk_probability'] = predictions[valid_coords]
            map_df['risk_category'] = map_df['risk_probability'].apply(
                lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low'
            )
            
            # Create borough-colored map if borough data available
            if 'borough' in map_df.columns:
                fig = px.scatter_mapbox(
                    map_df,
                    lat='latitude',
                    lon='longitude',
                    color='borough',
                    size='risk_probability',
                    hover_data={
                        'borough': True,
                        'risk_probability': ':.3f',
                        'risk_category': True,
                        'latitude': ':.4f',
                        'longitude': ':.4f'
                    },
                    title='NYC Office Building Risk by Borough',
                    zoom=9,
                    height=600,
                    color_discrete_map={
                        'Manhattan': '#003C7D',
                        'Brooklyn': '#0052A5', 
                        'Queens': '#FFB81C',
                        'Bronx': '#002855',
                        'Staten Island': '#6699CC'
                    }
                )
            else:
                # Fallback to risk-colored map
                fig = px.scatter_mapbox(
                    map_df,
                    lat='latitude',
                    lon='longitude',
                    color='risk_probability',
                    size='bldgarea' if 'bldgarea' in df.columns else None,
                    color_continuous_scale='RdYlGn_r',
                    title='NYC Office Building Vacancy Risk Map',
                    zoom=10,
                    height=600,
                    hover_data=['borough', 'risk_probability'] if 'borough' in df.columns else ['risk_probability']
                )
            
            fig.update_layout(mapbox_style="open-street-map")
            return fig
    
    return None

def main():
    """Main dashboard application."""
    
    # Stunning Header with Animation
    st.markdown('<h1 class="main-header">🏢 Office Apocalypse Algorithm</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered NYC Office Building Vacancy Risk Prediction System</p>', unsafe_allow_html=True)
    
    # Champion Badge with Animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div style="text-align: center;"><div class="champion-badge">🏆 Champion Model: XGBoost | 92.41% ROC-AUC | 93.01% Precision@10%</div></div>', 
            unsafe_allow_html=True
        )
    
    # Team Information
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="team-badge">👨‍🎓 Ibrahim Denis Fofanah (Lead)</span>
        <span class="team-badge">👨‍🎓 Bright Arowny Zaman</span>
        <span class="team-badge">👨‍🎓 Jeevan Hemanth Yendluri</span>
        <br><br>
        <span style="color: #003C7D; font-size: 1rem; font-weight: 600;">
            📚 <strong>PACE University</strong> | Data Science Capstone Project | 
            👨‍🏫 Advisor: Dr. Krishna Bathula | 📅 Fall 2025
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #003C7D 0%, #0052A5 100%); 
                padding: 1.5rem; 
                border-radius: 1rem; 
                margin: 2rem 0;
                box-shadow: 0 4px 15px rgba(0, 60, 125, 0.3);">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center;">
            <div style="color: white; padding: 0.5rem;">
                <h3 style="color: #FFB81C; margin: 0; font-size: 2rem; font-weight: 700;">7,191</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">NYC Office Buildings</p>
            </div>
            <div style="color: white; padding: 0.5rem;">
                <h3 style="color: #FFB81C; margin: 0; font-size: 2rem; font-weight: 700;">36</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Predictive Features</p>
            </div>
            <div style="color: white; padding: 0.5rem;">
                <h3 style="color: #FFB81C; margin: 0; font-size: 2rem; font-weight: 700;">6</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Integrated Datasets</p>
            </div>
            <div style="color: white; padding: 0.5rem;">
                <h3 style="color: #FFB81C; margin: 0; font-size: 2rem; font-weight: 700;">3.1×</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Efficiency Improvement</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Debug info (can be removed later)
    with st.expander("🔧 System Information", expanded=False):
        st.write(f"**Script Directory:** {script_dir}")
        st.write(f"**Project Root:** {project_root}")
        st.write(f"**Data Path:** {os.path.join(project_root, 'data', 'processed', 'office_buildings_clean.csv')}")
        st.write(f"**Model Path:** {os.path.join(project_root, 'models', 'champion_xgboost.pkl')}")
        st.write(f"**Data File Exists:** {os.path.exists(os.path.join(project_root, 'data', 'processed', 'office_buildings_clean.csv'))}")
        st.write(f"**Model File Exists:** {os.path.exists(os.path.join(project_root, 'models', 'champion_xgboost.pkl'))}")
    
    # Load data and model
    with st.spinner('🔄 Loading champion model and data...'):
        df = load_data()
        model, feature_names = load_champion_model()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please check file paths.")
        return
    
    # Success message with style
    st.success(f"✅ Successfully loaded {len(df):,} NYC office buildings and champion XGBoost model")
    
    # Enhanced Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin: 0;">🧭 Navigation</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin-top: 0.5rem;">
            Select an analysis module
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose Analysis Module",
        ["🏠 Building Lookup", "📊 Risk Overview", "🗺️ Risk Map", "🎯 Intervention Planning"],
        label_visibility="collapsed"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color: white; padding: 1.2rem; background: rgba(255,184,28,0.15); border-radius: 0.8rem; border: 2px solid rgba(255,184,28,0.3);">
        <h4 style="color: #FFB81C; margin-top: 0; font-weight: 700;">🏆 Model Performance</h4>
        <ul style="font-size: 0.95rem; line-height: 2; list-style: none; padding-left: 0;">
            <li>✓ <strong>ROC-AUC:</strong> 92.41%</li>
            <li>✓ <strong>Accuracy:</strong> 87.62%</li>
            <li>✓ <strong>Precision@10%:</strong> 93.01%</li>
            <li>✓ <strong>Business Impact:</strong> 3.1× efficiency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color: white; padding: 1rem;">
        <h4 style="color: #FFB81C; margin-top: 0; font-weight: 700;">ℹ️ About This Project</h4>
        <p style="font-size: 0.9rem; line-height: 1.6;">
            This dashboard provides <strong>AI-powered, building-level vacancy risk predictions</strong> 
            for NYC office properties using advanced machine learning.
        </p>
        <p style="font-size: 0.85rem; line-height: 1.6; margin-top: 1rem;">
            <strong style="color: #FFB81C;">📊 Data Sources (6):</strong><br>
            • NYC PLUTO (Building Data)<br>
            • ACRIS (Property Transactions)<br>
            • MTA Ridership (Transportation)<br>
            • DOB Permits (Construction)<br>
            • Business Registry<br>
            • Storefronts Vacancy
        </p>
        <p style="font-size: 0.85rem; line-height: 1.6; margin-top: 1rem;">
            <strong style="color: #FFB81C;">🎯 Use Cases:</strong><br>
            • Policy intervention planning<br>
            • Investment risk assessment<br>
            • Urban planning insights<br>
            • Economic development strategy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "🏠 Building Lookup":
        building_lookup_page(df, model, feature_names)
    elif page == "📊 Risk Overview":
        risk_overview_page(df, model, feature_names)
    elif page == "🗺️ Risk Map":
        risk_map_page(df, model, feature_names)
    elif page == "🎯 Intervention Planning":
        intervention_planning_page(df, model, feature_names)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="color: #003C7D; margin-bottom: 1rem; font-weight: 700;">Office Apocalypse Algorithm</h3>
        <p style="margin: 0.5rem 0; color: #002855; font-size: 1rem;">
            <strong>Team:</strong> Ibrahim Denis Fofanah (Lead), Bright Arowny Zaman, Jeevan Hemanth Yendluri
        </p>
        <p style="margin: 0.5rem 0; color: #002855; font-size: 1rem;">
            <strong>Institution:</strong> PACE University | <strong>Advisor:</strong> Dr. Krishna Bathula
        </p>
        <p style="margin: 0.5rem 0; color: #002855; font-size: 1rem;">
            Data Science Capstone Project | Fall 2025
        </p>
        <p style="margin-top: 1rem; font-size: 0.95rem; color: #495057; font-weight: 500;">
            © 2025 Office Apocalypse Algorithm Team | All Rights Reserved
        </p>
        <p style="margin-top: 0.5rem; font-size: 0.95rem;">
            🔗 <a href="https://github.com/Denis060/capstone_office-apocalypse-algorithm" target="_blank" style="color: #003C7D; font-weight: 600; text-decoration: none;">GitHub Repository</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def building_lookup_page(df, model, feature_names):
    """Building-specific risk analysis page."""
    st.markdown("""
    <h2 style='color: #003C7D; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;'>
        🏠 Individual Building Risk Analysis
    </h2>
    <p style='color: #002855; font-size: 1.1rem; margin-bottom: 1rem;'>
        Search and analyze vacancy risk for specific NYC office buildings with AI-powered predictions and SHAP explanations.
    </p>
    """, unsafe_allow_html=True)
    
    # How to Use Guide
    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        <div style="padding: 1rem; background: white;">
            <h4 style="color: #003C7D; font-size: 1.2rem; margin-bottom: 1rem;">🎯 Step-by-Step Guide:</h4>
            <ol style="line-height: 2.2; font-size: 1rem; color: #2c3e50;">
                <li><strong style="color: #003C7D;">Select a search method</strong> (by index, address, or borough)</li>
                <li><strong style="color: #003C7D;">Choose a building</strong> from the dropdown menu</li>
                <li><strong style="color: #003C7D;">Review the risk score</strong> displayed in the gauge (0-100%)</li>
                <li><strong style="color: #003C7D;">Examine building characteristics</strong> in the metrics below</li>
                <li><strong style="color: #003C7D;">Understand the prediction</strong> using the SHAP explanation chart</li>
            </ol>
            <div style="margin-top: 1.5rem; padding: 1rem; background: #fff8e1; border-left: 4px solid #FFB81C; border-radius: 0.5rem;">
                <p style="margin: 0; font-size: 1rem; color: #2c3e50; line-height: 1.6;">
                    <strong style="color: #003C7D; font-size: 1.1rem;">💡 Pro Tip:</strong><br>
                    Red bars in the SHAP chart push predictions toward <strong>HIGH risk</strong>, 
                    while blue bars push toward <strong>LOW risk</strong>. Longer bars = stronger influence.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Building selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search options
        search_method = st.selectbox(
            "Search Method",
            ["Building Index", "Address (if available)", "Borough Filter"]
        )
        
        if search_method == "Building Index":
            building_idx = st.selectbox(
                "Select Building",
                range(len(df)),
                format_func=lambda x: (
                    f"Building {x+1} - {df.iloc[x]['address']}" if 'address' in df.columns and pd.notna(df.iloc[x]['address']) and df.iloc[x]['address'] != 'No Address'
                    else f"Building {x+1} - BBL: {df.iloc[x].get('BBL', 'Unknown')} ({df.iloc[x].get('borough', 'Unknown')})"
                )
            )
        elif search_method == "Borough Filter" and 'borough' in df.columns:
            borough = st.selectbox("Select Borough", df['borough'].unique())
            borough_buildings = df[df['borough'] == borough]
            if len(borough_buildings) > 0:
                building_idx = st.selectbox(
                    "Select Building",
                    borough_buildings.index,
                    format_func=lambda x: (
                        f"Building {x+1} - {borough_buildings.loc[x, 'address']}" if 'address' in df.columns and pd.notna(borough_buildings.loc[x, 'address']) and borough_buildings.loc[x, 'address'] != 'No Address'
                        else f"Building {x+1} - BBL: {borough_buildings.loc[x].get('BBL', 'Unknown')}"
                    )
                )
            else:
                st.warning("No buildings found in selected borough")
                return
        else:
            building_idx = st.selectbox("Select Building", range(min(100, len(df))))
    
    # Get building data
    building = df.iloc[building_idx]
    
    # Prepare features for prediction
    building_features = []
    for feature in feature_names:
        if feature in building.index:
            value = building[feature]
            # Handle various data types and edge cases
            try:
                # Convert to string first, then clean
                value_str = str(value).strip()
                
                # Remove brackets if present
                if value_str.startswith('[') and value_str.endswith(']'):
                    value_str = value_str[1:-1].strip()
                
                # Handle empty strings
                if value_str == '' or value_str.lower() == 'nan':
                    value = 0.0
                else:
                    value = float(value_str)
                    
                # Check for inf/nan after conversion
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
            except (ValueError, TypeError, AttributeError):
                value = 0.0
            building_features.append(value)
        else:
            building_features.append(0.0)
    
    building_features = np.array(building_features, dtype=np.float64).reshape(1, -1)
    
    # Make prediction
    try:
        risk_probability = model.predict_proba(building_features)[0, 1]
        risk_label, risk_class = get_risk_category(risk_probability)
        
        # Display results
        with col2:
            st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Building ID", f"#{building_idx + 1}")
            if 'address' in building.index and pd.notna(building['address']):
                st.write(f"**Address:** {building['address']}")
            if 'borough' in building.index and pd.notna(building['borough']):
                st.write(f"**Borough:** {building['borough']}")
            st.markdown(f'<div class="{risk_class}">Risk Level: {risk_label}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk gauge
        st.plotly_chart(create_risk_gauge(risk_probability), use_container_width=True)
        
        # Building characteristics
        st.subheader("Building Characteristics")
        char_col1, char_col2, char_col3 = st.columns(3)
        
        with char_col1:
            if 'building_age' in building.index:
                st.metric("Building Age", f"{building['building_age']:.0f} years")
            if 'officearea' in building.index:
                st.metric("Office Area", f"{building['officearea']:,.0f} sq ft")
        
        with char_col2:
            if 'numfloors' in building.index:
                st.metric("Number of Floors", f"{building['numfloors']:.0f}")
            if 'assesstot' in building.index:
                st.metric("Assessed Value", f"${building['assesstot']:,.0f}")
        
        with char_col3:
            if 'office_ratio' in building.index:
                st.metric("Office Ratio", f"{building['office_ratio']:.2%}")
            if 'value_per_sqft' in building.index:
                st.metric("Value per Sq Ft", f"${building['value_per_sqft']:.0f}")
        
        # Feature Summary
        st.subheader("Key Building Features")
        
        # Display top 10 features in a clean table format
        feature_data = []
        for i, fname in enumerate(feature_names[:10]):
            try:
                fval = building_features[0][i]
                if not (pd.isna(fval) or np.isinf(fval)):
                    feature_data.append({
                        'Feature': fname.replace('_', ' ').title(),
                        'Value': f"{float(fval):.2f}" if abs(float(fval)) < 1000 else f"{float(fval):,.0f}"
                    })
            except:
                pass
        
        if feature_data:
            st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)
        
        st.info("💡 **Model Insight:** This building's risk prediction is based on 20 engineered features including building age, office area, construction activity, and market indicators. The XGBoost model achieved 92.41% ROC-AUC on validation data.")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")

def risk_overview_page(df, model, feature_names):
    """Portfolio risk overview page."""
    st.markdown("""
    <h2 style='color: #003C7D; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;'>
        📊 Portfolio Risk Overview
    </h2>
    <p style='color: #002855; font-size: 1.1rem; margin-bottom: 2rem;'>
        Comprehensive analysis of vacancy risk across all {count:,} NYC office buildings in the portfolio.
    </p>
    """.format(count=len(df)), unsafe_allow_html=True)
    
    # Make predictions for all buildings
    with st.spinner('Analyzing all buildings...'):
        # Prepare all features
        all_features = []
        for _, building in df.iterrows():
            building_features = []
            for feature in feature_names:
                if feature in building.index:
                    value = building[feature]
                    if pd.isna(value) or np.isinf(value):
                        value = 0
                    building_features.append(value)
                else:
                    building_features.append(0)
            all_features.append(building_features)
        
        all_features = np.array(all_features)
        predictions = model.predict_proba(all_features)[:, 1]
    
    # Summary metrics
    high_risk_count = sum(predictions >= 0.7)
    medium_risk_count = sum((predictions >= 0.4) & (predictions < 0.7))
    low_risk_count = sum(predictions < 0.4)
    avg_risk = predictions.mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Buildings", f"{len(df):,}")
    with col2:
        st.metric("🔴 High Risk", f"{high_risk_count:,}", f"{high_risk_count/len(df):.1%}")
    with col3:
        st.metric("🟡 Medium Risk", f"{medium_risk_count:,}", f"{medium_risk_count/len(df):.1%}")
    with col4:
        st.metric("🟢 Low Risk", f"{low_risk_count:,}", f"{low_risk_count/len(df):.1%}")
    
    # Key Insights Box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #003C7D 0%, #0052A5 100%); 
                padding: 1.5rem; 
                border-radius: 1rem; 
                margin: 1.5rem 0;
                color: white;
                box-shadow: 0 4px 15px rgba(0, 60, 125, 0.3);">
        <h4 style="color: #FFB81C; margin-top: 0;">🔍 Key Portfolio Insights</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div>
                <strong style="color: #FFB81C;">Average Risk:</strong><br>
                {avg_risk:.1%} across all buildings
            </div>
            <div>
                <strong style="color: #FFB81C;">High Priority:</strong><br>
                {high_risk_count:,} buildings need immediate attention
            </div>
            <div>
                <strong style="color: #FFB81C;">Risk Concentration:</strong><br>
                {concentration:.1%} of portfolio is high/medium risk
            </div>
            <div>
                <strong style="color: #FFB81C;">Model Confidence:</strong><br>
                92.41% ROC-AUC, 87.62% Accuracy
            </div>
        </div>
    </div>
    """.format(
        avg_risk=avg_risk,
        high_risk_count=high_risk_count,
        concentration=(high_risk_count + medium_risk_count) / len(df)
    ), unsafe_allow_html=True)
    
    # Risk distribution
    st.subheader("📊 Risk Distribution Across Boroughs")
    fig = create_risk_distribution_plot(df, predictions)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top risk buildings
    st.subheader("🚨 Highest Risk Buildings")
    risk_df = df.copy()
    risk_df['risk_probability'] = predictions
    risk_df['risk_rank'] = risk_df['risk_probability'].rank(ascending=False)
    
    top_risk = risk_df.nlargest(20, 'risk_probability')
    
    display_cols = ['risk_rank', 'risk_probability']
    if 'address' in df.columns:
        display_cols.append('address')
    if 'borough' in df.columns:
        display_cols.append('borough')
    display_cols.extend(['building_age', 'officearea', 'assesstot'])
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in top_risk.columns]
    
    st.dataframe(
        top_risk[available_cols].round(3),
        use_container_width=True
    )

def risk_map_page(df, model, feature_names):
    """Geographic risk visualization page."""
    st.markdown("""
    <h2 style='color: #003C7D; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;'>
        🗺️ Geographic Risk Distribution
    </h2>
    <p style='color: #002855; font-size: 1.1rem; margin-bottom: 2rem;'>
        Interactive map showing spatial patterns of office building vacancy risk across NYC boroughs.
    </p>
    """, unsafe_allow_html=True)
    
    # Make predictions
    with st.spinner('Generating risk predictions...'):
        all_features = []
        for _, building in df.iterrows():
            building_features = []
            for feature in feature_names:
                if feature in building.index:
                    value = building[feature]
                    if pd.isna(value) or np.isinf(value):
                        value = 0
                    building_features.append(value)
                else:
                    building_features.append(0)
            all_features.append(building_features)
        
        all_features = np.array(all_features)
        predictions = model.predict_proba(all_features)[:, 1]
    
    # Create map
    fig = create_geographic_risk_map(df, predictions)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # Map legend and info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: white; 
                        padding: 1.5rem; 
                        border-radius: 0.8rem; 
                        border: 3px solid #003C7D;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4 style="color: #003C7D; margin-top: 0; font-size: 1.2rem; font-weight: 700;">🗺️ Borough Map Legend</h4>
                <ul style="line-height: 2.2; font-size: 1rem; color: #2c3e50; list-style: none; padding-left: 0;">
                    <li><span style="color: #003C7D; font-size: 1.5rem;">●</span> <strong>Manhattan:</strong> Central business district</li>
                    <li><span style="color: #0052A5; font-size: 1.5rem;">●</span> <strong>Brooklyn:</strong> Southern borough</li>
                    <li><span style="color: #FFB81C; font-size: 1.5rem;">●</span> <strong>Queens:</strong> Eastern borough</li>
                    <li><span style="color: #002855; font-size: 1.5rem;">●</span> <strong>Bronx:</strong> Northern borough</li>
                    <li><span style="color: #6699CC; font-size: 1.5rem;">●</span> <strong>Staten Island:</strong> Southwest borough</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; 
                        padding: 1.5rem; 
                        border-radius: 0.8rem; 
                        border: 3px solid #FFB81C;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4 style="color: #003C7D; margin-top: 0; font-size: 1.2rem; font-weight: 700;">📍 Risk Indicators</h4>
                <ul style="line-height: 2.2; font-size: 1rem; color: #2c3e50;">
                    <li><strong>Point size</strong> = Risk probability magnitude</li>
                    <li><strong>Hover</strong> for detailed building information</li>
                    <li><strong>Colors</strong> show borough-specific patterns</li>
                    <li><strong>Clustering</strong> indicates concentrated risk areas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Borough risk analysis
        if 'borough' in df.columns:
            st.subheader("📊 Risk Analysis by Borough")
            
            risk_df = df.copy()
            risk_df['risk_probability'] = predictions
            risk_df['high_risk'] = (risk_df['risk_probability'] >= 0.7).astype(int)
            
            borough_stats = risk_df.groupby('borough').agg({
                'risk_probability': ['mean', 'std', 'count'],
                'high_risk': 'sum',
                'officearea': 'sum' if 'officearea' in df.columns else 'count',
                'assesstot': 'mean' if 'assesstot' in df.columns else 'count'
            }).round(3)
            
            # Flatten column names
            borough_stats.columns = [
                'Avg_Risk', 'Risk_StdDev', 'Building_Count', 
                'High_Risk_Count', 'Total_Office_Area', 'Avg_Assessment'
            ]
            
            # Add risk percentage
            borough_stats['High_Risk_Rate'] = (
                borough_stats['High_Risk_Count'] / borough_stats['Building_Count']
            ).round(3)
            
            # Sort by risk
            borough_stats = borough_stats.sort_values('Avg_Risk', ascending=False)
            
            st.dataframe(borough_stats, use_container_width=True)
            
            # Key insights
            highest_risk_borough = borough_stats.index[0]
            lowest_risk_borough = borough_stats.index[-1]
            
            st.markdown(f"""
            **🔍 Geographic Insights:**
            - **Highest Risk:** {highest_risk_borough} ({borough_stats.loc[highest_risk_borough, 'Avg_Risk']:.1%} avg risk)
            - **Lowest Risk:** {lowest_risk_borough} ({borough_stats.loc[lowest_risk_borough, 'Avg_Risk']:.1%} avg risk)
            - **Most Buildings:** {borough_stats['Building_Count'].idxmax()} ({borough_stats['Building_Count'].max():,} buildings)
            - **Total Portfolio:** {borough_stats['Building_Count'].sum():,} buildings analyzed
            """)
    
    else:
        st.warning("Geographic coordinates not available for mapping.")
        
        # Alternative: Borough-based analysis
        if 'borough' in df.columns:
            st.subheader("📍 Risk Analysis by Borough")
            risk_df = df.copy()
            risk_df['risk_probability'] = predictions
            
            borough_stats = risk_df.groupby('borough').agg({
                'risk_probability': ['mean', 'count'],
                'officearea': 'sum' if 'officearea' in df.columns else 'count'
            }).round(3)
            
            borough_stats.columns = ['Avg_Risk', 'Building_Count', 'Total_Office_Area']
            borough_stats = borough_stats.sort_values('Avg_Risk', ascending=False)
            
            # Create borough risk chart
            fig_borough = px.bar(
                x=borough_stats.index,
                y=borough_stats['Avg_Risk'],
                title='Average Vacancy Risk by Borough',
                labels={'x': 'Borough', 'y': 'Average Risk Probability'},
                color=borough_stats['Avg_Risk'],
                color_continuous_scale='RdYlGn_r'
            )
            
            st.plotly_chart(fig_borough, use_container_width=True)
            st.dataframe(borough_stats, use_container_width=True)

def intervention_planning_page(df, model, feature_names):
    """Intervention prioritization page."""
    st.markdown("""
    <h2 style='color: #003C7D; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;'>
        🎯 Intervention Planning & Targeting
    </h2>
    <p style='color: #002855; font-size: 1.1rem; margin-bottom: 2rem;'>
        Prioritize building interventions based on risk scores, building characteristics, and policy parameters.
    </p>
    """, unsafe_allow_html=True)
    
    # Make predictions
    with st.spinner('Prioritizing interventions...'):
        all_features = []
        for _, building in df.iterrows():
            building_features = []
            for feature in feature_names:
                if feature in building.index:
                    value = building[feature]
                    if pd.isna(value) or np.isinf(value):
                        value = 0
                    building_features.append(value)
                else:
                    building_features.append(0)
            all_features.append(building_features)
        
        all_features = np.array(all_features)
        predictions = model.predict_proba(all_features)[:, 1]
    
    # Intervention parameters
    st.subheader("Targeting Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        max_interventions = st.slider("Max Interventions", 10, 500, 100, 10)
    
    with col2:
        if 'officearea' in df.columns:
            min_size = st.slider("Min Building Size (sq ft)", 
                                int(df['officearea'].min()), 
                                int(df['officearea'].max()), 
                                int(df['officearea'].median()))
        else:
            min_size = 0
    
    # Filter buildings for intervention
    risk_df = df.copy()
    risk_df['risk_probability'] = predictions
    
    # Apply filters
    intervention_candidates = risk_df[
        (risk_df['risk_probability'] >= risk_threshold)
    ]
    
    if 'officearea' in df.columns:
        intervention_candidates = intervention_candidates[
            intervention_candidates['officearea'] >= min_size
        ]
    
    # Sort by risk and limit
    intervention_candidates = intervention_candidates.nlargest(max_interventions, 'risk_probability')
    
    # Results
    st.subheader(f"🎯 {len(intervention_candidates)} Buildings Identified for Intervention")
    
    if len(intervention_candidates) > 0:
        # Summary stats
        total_office_area = intervention_candidates['officearea'].sum() if 'officearea' in df.columns else 0
        avg_risk = intervention_candidates['risk_probability'].mean()
        
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric("Buildings Targeted", len(intervention_candidates))
        with met_col2:
            st.metric("Average Risk", f"{avg_risk:.1%}")
        with met_col3:
            if total_office_area > 0:
                st.metric("Total Office Area", f"{total_office_area:,.0f} sq ft")
        
        # Intervention list
        display_cols = ['risk_probability']
        if 'address' in df.columns:
            display_cols.append('address')
        if 'borough' in df.columns:
            display_cols.append('borough')
        if 'building_age' in df.columns:
            display_cols.append('building_age')
        if 'officearea' in df.columns:
            display_cols.append('officearea')
        if 'assesstot' in df.columns:
            display_cols.append('assesstot')
        
        available_cols = [col for col in display_cols if col in intervention_candidates.columns]
        
        st.dataframe(
            intervention_candidates[available_cols].round(3),
            use_container_width=True
        )
        
        # Export option
        csv = intervention_candidates.to_csv(index=False)
        st.download_button(
            label="📁 Download Intervention List (CSV)",
            data=csv,
            file_name=f"intervention_list_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No buildings meet the current criteria. Try adjusting the filters.")

if __name__ == "__main__":
    main()