"""
Data Leakage Analysis Script

This script identifies features that are causing data leakage in our model.
"""

import pandas as pd
import numpy as np

def analyze_data_leakage():
    """Analyze potential data leakage in the processed dataset."""
    print("DATA LEAKAGE ANALYSIS - OFFICE APOCALYPSE ALGORITHM")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/processed/office_buildings_processed.csv')
    print(f"Loaded data: {len(data):,} records")
    
    # 1. Direct leakage features
    print("\\n1. DIRECT LEAKAGE FEATURES (should NOT be in training):")
    print("-" * 50)
    direct_leakage = [
        'vacancy_risk_alert',
        'vacancy_risk_early_warning', 
        'target_high_vacancy_risk'
    ]
    
    for feat in direct_leakage:
        if feat in data.columns:
            if feat == 'target_high_vacancy_risk':
                corr = 1.0
            elif feat == 'vacancy_risk_alert':
                # Check perfect correlation through crosstab
                crosstab = pd.crosstab(data['target_high_vacancy_risk'], data[feat])
                print(f"  {feat}: CATEGORICAL LEAKAGE")
                print(f"    Crosstab:\\n{crosstab}")
                continue
            else:
                corr = data[feat].corr(data['target_high_vacancy_risk'])
            print(f"  {feat}: correlation = {corr:.4f}")
    
    # 2. Risk composite features (likely contain leakage)
    print("\\n2. RISK COMPOSITE FEATURES (likely contain leakage):")
    print("-" * 50)
    risk_features = [col for col in data.columns if 'risk' in col.lower()]
    for feat in risk_features:
        if feat not in direct_leakage:
            corr = data[feat].corr(data['target_high_vacancy_risk'])
            print(f"  {feat}: correlation = {corr:.4f}")
    
    # 3. Investment/potential features (likely derived from target)
    print("\\n3. INVESTMENT/POTENTIAL FEATURES (may contain leakage):")
    print("-" * 50)
    investment_features = [col for col in data.columns 
                          if any(word in col.lower() for word in ['investment', 'competitive', 'potential'])]
    for feat in investment_features:
        if data[feat].dtype in ['object', 'category']:
            print(f"  {feat}: CATEGORICAL - skipping correlation")
        else:
            corr = data[feat].corr(data['target_high_vacancy_risk'])
            print(f"  {feat}: correlation = {corr:.4f}")
    
    # 4. Safe features (raw data, not derived from target)
    print("\\n4. SAFE FEATURES (raw building data):")
    print("-" * 40)
    safe_features = [
        'building_age', 'lotarea', 'bldgarea', 'officearea', 'numfloors',
        'assessland', 'assesstot', 'yearbuilt', 'value_per_sqft',
        'office_ratio', 'floor_efficiency', 'transaction_count',
        'deed_count', 'mortgage_count'
    ]
    
    available_safe = [feat for feat in safe_features if feat in data.columns]
    print(f"Available safe features: {len(available_safe)}")
    for feat in available_safe:
        corr = data[feat].corr(data['target_high_vacancy_risk'])
        print(f"  {feat}: correlation = {corr:.4f}")
    
    # 5. Create clean dataset without leakage
    print("\\n5. CREATING CLEAN DATASET:")
    print("-" * 30)
    
    # Features to exclude (high leakage risk)
    exclude_features = [
        # Direct leakage
        'vacancy_risk_alert', 'vacancy_risk_early_warning', 'target_high_vacancy_risk',
        # Risk composites (likely contain leakage)
        'neighborhood_vacancy_risk', 'investment_risk', 'competitive_risk',
        # Investment scores (derived from target)
        'investment_potential_score', 'market_competitiveness_score',
        'modernization_potential', 'building_quality',
        # Economic composites (may contain leakage)
        'economic_distress_composite', 'neighborhood_vitality_index',
        # Category features (derived from scores)
        'economic_distress_category', 'investment_potential_category',
        'market_competitiveness_category', 'neighborhood_vitality_category',
        # Other derived features
        'office_size_advantage', 'mixed_use_advantage', 'location_advantage'
    ]
    
    # Keep only safe features
    safe_feature_set = [
        # Building characteristics (raw data)
        'building_age', 'office_ratio', 'floor_efficiency', 'value_per_sqft',
        'land_value_ratio', 'lotarea', 'bldgarea', 'officearea', 'numfloors',
        'assessland', 'assesstot', 'yearbuilt',
        
        # Transaction data (raw counts)
        'transaction_count', 'deed_count', 'mortgage_count',
        
        # Basic geographic/accessibility (if not derived from target)
        'mta_accessibility_proxy', 'business_density_proxy', 'construction_activity_proxy',
        
        # Basic ratios and interactions
        'commercial_ratio', 'distress_score'
    ]
    
    # Filter to available columns
    available_safe_features = [feat for feat in safe_feature_set if feat in data.columns]
    excluded_count = len([feat for feat in exclude_features if feat in data.columns])
    
    print(f"Features to exclude: {excluded_count}")
    print(f"Safe features to keep: {len(available_safe_features)}")
    
    # Create clean dataset
    clean_features = available_safe_features + ['target_high_vacancy_risk', 'BBL']
    clean_data = data[clean_features].copy()
    
    print(f"\\nClean dataset shape: {clean_data.shape}")
    print(f"Features in clean dataset: {len(available_safe_features)}")
    
    # Check correlations in clean dataset
    print("\\n6. CORRELATIONS IN CLEAN DATASET:")
    print("-" * 35)
    target_corr = clean_data.select_dtypes(include=[np.number]).corrwith(
        clean_data['target_high_vacancy_risk']
    ).abs().sort_values(ascending=False)
    
    print("Top correlations with target (should be much lower):")
    for feat, corr in target_corr.head(10).items():
        if feat != 'target_high_vacancy_risk':
            print(f"  {feat}: {corr:.4f}")
    
    # Save clean dataset
    clean_data.to_csv('data/processed/office_buildings_clean.csv', index=False)
    print(f"\\nSaved clean dataset: data/processed/office_buildings_clean.csv")
    
    return clean_data, available_safe_features

if __name__ == "__main__":
    clean_data, safe_features = analyze_data_leakage()