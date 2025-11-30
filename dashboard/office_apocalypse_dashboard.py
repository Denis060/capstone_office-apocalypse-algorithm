#!/usr/bin/env python3
"""
Office Apocalypse Algorithm - Interactive Dashboard
Streamlit web application for vacancy risk prediction

Champion Model: XGBoost (92.41% ROC-AUC)
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
    page_title="Office Apocalypse Algorithm",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e9ef;
    }
    .high-risk {
        color: #ff4444;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffaa00;
        font-weight: bold;
    }
    .low-risk {
        color: #44ff44;
        font-weight: bold;
    }
    .champion-badge {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #000;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
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
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Vacancy Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_plot(shap_values, feature_names, building_data):
    """Create SHAP feature importance plot for a specific building."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get SHAP values for this building
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Sort features by absolute SHAP value
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'feature_value': building_data
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
        ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                f'{feature_val:.2f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_risk_distribution_plot(df, predictions):
    """Create risk distribution visualization."""
    risk_df = pd.DataFrame({
        'Probability': predictions,
        'Borough': df['borough'] if 'borough' in df.columns else 'Unknown'
    })
    
    fig = px.histogram(
        risk_df, 
        x='Probability', 
        color='Borough',
        title='Vacancy Risk Distribution Across NYC',
        nbins=30,
        labels={'Probability': 'Vacancy Risk Probability', 'count': 'Number of Buildings'}
    )
    
    # Add risk threshold lines
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    fig.update_layout(height=400)
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
                        'Manhattan': '#FF6B6B',
                        'Brooklyn': '#4ECDC4', 
                        'Queens': '#45B7D1',
                        'Bronx': '#96CEB4',
                        'Staten Island': '#FFEAA7'
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
    
    # Header
    st.markdown('<h1 class="main-header">🏢 Office Apocalypse Algorithm</h1>', unsafe_allow_html=True)
    st.markdown('<div class="champion-badge">🏆 Champion Model: XGBoost (92.41% ROC-AUC)</div>', unsafe_allow_html=True)
    
    # Debug info (can be removed later)
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"**Script Directory:** {script_dir}")
        st.write(f"**Project Root:** {project_root}")
        st.write(f"**Data Path:** {os.path.join(project_root, 'data', 'processed', 'office_buildings_clean.csv')}")
        st.write(f"**Model Path:** {os.path.join(project_root, 'models', 'champion_xgboost.pkl')}")
        st.write(f"**Data File Exists:** {os.path.exists(os.path.join(project_root, 'data', 'processed', 'office_buildings_clean.csv'))}")
        st.write(f"**Model File Exists:** {os.path.exists(os.path.join(project_root, 'models', 'champion_xgboost.pkl'))}")
    
    # Load data and model
    with st.spinner('Loading champion model and data...'):
        df = load_data()
        model, feature_names = load_champion_model()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please check file paths.")
        return
    
    st.success(f"✅ Loaded {len(df):,} NYC office buildings and champion XGBoost model")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["🏠 Building Lookup", "📊 Risk Overview", "🗺️ Risk Map", "🎯 Intervention Planning"]
    )
    
    if page == "🏠 Building Lookup":
        building_lookup_page(df, model, feature_names)
    elif page == "📊 Risk Overview":
        risk_overview_page(df, model, feature_names)
    elif page == "🗺️ Risk Map":
        risk_map_page(df, model, feature_names)
    elif page == "🎯 Intervention Planning":
        intervention_planning_page(df, model, feature_names)

def building_lookup_page(df, model, feature_names):
    """Building-specific risk analysis page."""
    st.header("🏠 Individual Building Risk Analysis")
    
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
                format_func=lambda x: f"Building {x+1} - {df.iloc[x]['address'] if 'address' in df.columns else 'No Address'}"
            )
        elif search_method == "Borough Filter" and 'borough' in df.columns:
            borough = st.selectbox("Select Borough", df['borough'].unique())
            borough_buildings = df[df['borough'] == borough]
            if len(borough_buildings) > 0:
                building_idx = st.selectbox(
                    "Select Building",
                    borough_buildings.index,
                    format_func=lambda x: f"Building {x+1} - {borough_buildings.loc[x, 'address'] if 'address' in df.columns else 'No Address'}"
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
            if pd.isna(value) or np.isinf(value):
                value = 0
            building_features.append(value)
        else:
            building_features.append(0)
    
    building_features = np.array(building_features).reshape(1, -1)
    
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
        
        # SHAP explanation
        st.subheader("Prediction Explanation")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(building_features)
            
            fig = create_feature_importance_plot(shap_values, feature_names, building_features[0])
            st.pyplot(fig)
            
            st.info("""
            **How to read this chart:**
            - Red bars push the prediction toward HIGH risk
            - Blue bars push the prediction toward LOW risk  
            - Longer bars = stronger influence on the prediction
            - Numbers show the actual feature values for this building
            """)
        except Exception as e:
            st.warning(f"Could not generate explanation: {e}")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")

def risk_overview_page(df, model, feature_names):
    """Portfolio risk overview page."""
    st.header("📊 Portfolio Risk Overview")
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Buildings", f"{len(df):,}")
    with col2:
        st.metric("🔴 High Risk", f"{high_risk_count:,}", f"{high_risk_count/len(df):.1%}")
    with col3:
        st.metric("🟡 Medium Risk", f"{medium_risk_count:,}", f"{medium_risk_count/len(df):.1%}")
    with col4:
        st.metric("🟢 Low Risk", f"{low_risk_count:,}", f"{low_risk_count/len(df):.1%}")
    
    # Risk distribution
    st.subheader("Risk Distribution")
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
    st.header("🗺️ Geographic Risk Distribution")
    
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
            st.info("""
            **Borough Map Legend:**
            - 🔴 Manhattan: Central business district
            - 🟦 Queens: Eastern borough 
            - 🟢 Bronx: Northern borough
            - 🔵 Brooklyn: Southern borough
            - 🟡 Staten Island: Southwest borough
            """)
        
        with col2:
            st.info("""
            **Risk Indicators:**
            - Point size = Risk probability
            - Hover for details
            - Colors show borough patterns
            - Clustering indicates geographic risk concentration
            """)
        
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
    st.header("🎯 Intervention Planning")
    
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