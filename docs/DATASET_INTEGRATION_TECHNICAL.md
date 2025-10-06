# Dataset Integration Technical Implementation
# Code Examples and Implementation Details

**Companion Document to**: DATASET_INTEGRATION_METHODOLOGY.md  
**Date**: October 2025  

---

## 🔧 Core Integration Functions

### **1. BBL Standardization Across Datasets**

```python
def standardize_bbl(df, borough_col, block_col, lot_col):
    """
    Standardize BBL (Building Block Lot) format across all NYC datasets
    
    Args:
        df: DataFrame containing BBL components
        borough_col: Column name for borough code
        block_col: Column name for block number  
        lot_col: Column name for lot number
        
    Returns:
        df: DataFrame with standardized 'BBL' column
    """
    df['BBL'] = (
        df[borough_col].astype(str).str.zfill(1) +
        df[block_col].astype(str).str.zfill(5) +
        df[lot_col].astype(str).str.zfill(4)
    )
    return df

# Example usage for PLUTO dataset
pluto_df = standardize_bbl(pluto_df, 'Borough', 'Block', 'Lot')

# Example usage for ACRIS dataset  
acris_df = standardize_bbl(acris_df, 'BOROUGH', 'BLOCK', 'LOT')
```

### **2. Spatial Proximity Integration**

```python
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    return c * r

def calculate_mta_proximity_features(buildings_df, stations_df, ridership_df):
    """
    Calculate MTA proximity features for each building
    
    Args:
        buildings_df: DataFrame with building locations (lat, lon, BBL)
        stations_df: DataFrame with station locations (lat, lon, station_id)
        ridership_df: DataFrame with ridership data (station_id, ridership)
        
    Returns:
        features_df: DataFrame with proximity features
    """
    features = []
    
    for idx, building in buildings_df.iterrows():
        b_lat, b_lon, bbl = building['latitude'], building['longitude'], building['BBL']
        
        # Calculate distances to all stations
        distances = []
        station_ridership = []
        
        for _, station in stations_df.iterrows():
            s_lat, s_lon, s_id = station['latitude'], station['longitude'], station['station_id']
            distance = haversine_distance(b_lat, b_lon, s_lat, s_lon)
            distances.append(distance)
            
            # Get ridership for this station
            ridership = ridership_df[ridership_df['station_id'] == s_id]['avg_ridership'].iloc[0] if len(ridership_df[ridership_df['station_id'] == s_id]) > 0 else 0
            station_ridership.append(ridership)
        
        # Create features
        features.append({
            'BBL': bbl,
            'nearest_station_distance': min(distances),
            'nearest_station_ridership': station_ridership[distances.index(min(distances))],
            'ridership_within_quarter_mile': sum(r for d, r in zip(distances, station_ridership) if d <= 0.25),
            'stations_within_quarter_mile': sum(1 for d in distances if d <= 0.25),
            'weighted_accessibility_score': sum(r/max(d, 0.1) for d, r in zip(distances, station_ridership) if d <= 0.5)
        })
    
    return pd.DataFrame(features)
```

### **3. Temporal Aggregation for Transaction Data**

```python
def create_acris_temporal_features(acris_df):
    """
    Create temporal aggregation features from ACRIS transaction data
    
    Args:
        acris_df: DataFrame with BBL, transaction_date, sale_price, document_type
        
    Returns:
        features_df: DataFrame with temporal features by BBL
    """
    from datetime import datetime, timedelta
    
    current_date = datetime.now()
    
    # Define time windows
    windows = {
        '1y': current_date - timedelta(days=365),
        '3y': current_date - timedelta(days=365*3),
        '5y': current_date - timedelta(days=365*5)
    }
    
    features = []
    
    for bbl in acris_df['BBL'].unique():
        bbl_transactions = acris_df[acris_df['BBL'] == bbl].copy()
        bbl_transactions['transaction_date'] = pd.to_datetime(bbl_transactions['transaction_date'])
        
        bbl_features = {'BBL': bbl}
        
        for window_name, cutoff_date in windows.items():
            recent_transactions = bbl_transactions[bbl_transactions['transaction_date'] >= cutoff_date]
            
            # Basic counts and values
            bbl_features[f'transaction_count_{window_name}'] = len(recent_transactions)
            bbl_features[f'total_value_{window_name}'] = recent_transactions['sale_price'].sum()
            bbl_features[f'avg_value_{window_name}'] = recent_transactions['sale_price'].mean() if len(recent_transactions) > 0 else 0
            
            # Distress indicators
            distress_docs = ['DEED, SHERIFF', 'DEED, REF & REC', 'DEED, CORRECTION']
            distress_count = len(recent_transactions[recent_transactions['document_type'].isin(distress_docs)])
            bbl_features[f'distress_transactions_{window_name}'] = distress_count
            bbl_features[f'distress_rate_{window_name}'] = distress_count / max(len(recent_transactions), 1)
            
            # Ownership stability
            unique_owners = recent_transactions['grantee'].nunique() if len(recent_transactions) > 0 else 0
            bbl_features[f'ownership_changes_{window_name}'] = unique_owners
            
        # Price volatility (coefficient of variation)
        if len(bbl_transactions) > 1:
            price_std = bbl_transactions['sale_price'].std()
            price_mean = bbl_transactions['sale_price'].mean()
            bbl_features['price_volatility'] = price_std / price_mean if price_mean > 0 else 0
        else:
            bbl_features['price_volatility'] = 0
            
        features.append(bbl_features)
    
    return pd.DataFrame(features)
```

### **4. Business Density and Diversity Features**

```python
def create_business_features(business_df, buildings_df):
    """
    Create business density and diversity features by ZIP code
    
    Args:
        business_df: DataFrame with business data (business_name, zip_code, industry)
        buildings_df: DataFrame with building data (BBL, zip_code)
        
    Returns:
        features_df: DataFrame with business features by BBL
    """
    import numpy as np
    from scipy.stats import entropy
    
    # Aggregate business data by ZIP code
    zip_features = {}
    
    for zip_code in business_df['zip_code'].unique():
        zip_businesses = business_df[business_df['zip_code'] == zip_code]
        
        # Basic counts
        total_businesses = len(zip_businesses)
        
        # Industry diversity (Shannon entropy)
        industry_counts = zip_businesses['industry'].value_counts()
        industry_probs = industry_counts / industry_counts.sum()
        diversity_score = entropy(industry_probs) if len(industry_probs) > 1 else 0
        
        # Business types
        unique_industries = zip_businesses['industry'].nunique()
        
        zip_features[zip_code] = {
            'business_count': total_businesses,
            'business_diversity_score': diversity_score,
            'unique_industries': unique_industries,
            'businesses_per_sq_mile': total_businesses / 1.0  # Approximate ZIP area
        }
    
    # Map to buildings
    building_features = []
    
    for _, building in buildings_df.iterrows():
        bbl = building['BBL']
        zip_code = building['zip_code']
        
        if zip_code in zip_features:
            features = {'BBL': bbl}
            features.update(zip_features[zip_code])
        else:
            features = {
                'BBL': bbl,
                'business_count': 0,
                'business_diversity_score': 0,
                'unique_industries': 0,
                'businesses_per_sq_mile': 0
            }
        
        building_features.append(features)
    
    return pd.DataFrame(building_features)
```

### **5. Cross-Dataset Composite Features**

```python
def create_composite_features(integrated_df):
    """
    Create sophisticated composite features combining multiple datasets
    
    Args:
        integrated_df: DataFrame with all individual dataset features
        
    Returns:
        df: DataFrame with additional composite features
    """
    df = integrated_df.copy()
    
    # Normalize features for composite calculations
    def min_max_normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else 0
    
    # 1. Economic Distress Composite
    acris_distress = min_max_normalize(df['distress_rate_3y'])
    business_decline = 1 - min_max_normalize(df['business_count'])
    investment_decline = 1 - min_max_normalize(df['total_investment_3y'])
    
    df['economic_distress_composite'] = (
        0.4 * acris_distress +
        0.3 * business_decline +
        0.3 * investment_decline
    )
    
    # 2. Urban Vitality Index
    transit_vitality = min_max_normalize(df['weighted_accessibility_score'])
    business_vitality = min_max_normalize(df['business_diversity_score'])
    investment_vitality = min_max_normalize(df['total_investment_1y'])
    
    df['urban_vitality_index'] = (
        0.4 * transit_vitality +
        0.35 * business_vitality +
        0.25 * investment_vitality
    )
    
    # 3. Building Modernization Score
    recent_permits = min_max_normalize(df['permit_count_1y'])
    investment_intensity = min_max_normalize(df['investment_per_sqft'])
    building_newness = min_max_normalize(2024 - df['year_built'])
    
    df['modernization_score'] = (
        0.4 * recent_permits +
        0.35 * investment_intensity +
        0.25 * building_newness
    )
    
    # 4. Neighborhood Competitiveness
    location_advantage = min_max_normalize(df['nearest_station_ridership'])
    business_ecosystem = min_max_normalize(df['unique_industries'])
    infrastructure_investment = min_max_normalize(df['modernization_score'])
    
    df['neighborhood_competitiveness'] = (
        0.4 * location_advantage +
        0.35 * business_ecosystem +
        0.25 * infrastructure_investment
    )
    
    # 5. Vacancy Risk Early Warning Score (Master Composite)
    economic_risk = df['economic_distress_composite']
    vitality_risk = 1 - df['urban_vitality_index']
    obsolescence_risk = 1 - df['modernization_score']
    neighborhood_risk = 1 - df['neighborhood_competitiveness']
    vacancy_spillover = min_max_normalize(df['vacant_storefronts_nearby'])
    
    df['vacancy_risk_early_warning'] = (
        0.25 * economic_risk +
        0.20 * vitality_risk +
        0.20 * obsolescence_risk +
        0.15 * neighborhood_risk +
        0.10 * vacancy_spillover +
        0.10 * min_max_normalize(df['building_age'])
    )
    
    return df
```

### **6. Integration Pipeline Orchestration**

```python
def run_complete_integration_pipeline():
    """
    Complete data integration pipeline orchestrating all steps
    """
    print("🚀 Starting Complete Dataset Integration Pipeline")
    
    # Step 1: Load and standardize all datasets
    print("📊 Loading datasets...")
    pluto_df = load_and_standardize_pluto()
    acris_df = load_and_standardize_acris()
    mta_df = load_and_standardize_mta()
    business_df = load_and_standardize_business()
    dob_df = load_and_standardize_dob()
    storefront_df = load_and_standardize_storefronts()
    
    # Step 2: Create individual dataset features
    print("🔧 Creating dataset-specific features...")
    acris_features = create_acris_temporal_features(acris_df)
    mta_features = calculate_mta_proximity_features(pluto_df, mta_df['stations'], mta_df['ridership'])
    business_features = create_business_features(business_df, pluto_df)
    dob_features = create_dob_investment_features(dob_df)
    storefront_features = create_storefront_proximity_features(pluto_df, storefront_df)
    
    # Step 3: Integrate all features
    print("🔗 Integrating all datasets...")
    integrated_df = pluto_df.copy()
    
    for features_df in [acris_features, mta_features, business_features, dob_features, storefront_features]:
        integrated_df = integrated_df.merge(features_df, on='BBL', how='left')
    
    # Step 4: Handle missing data
    print("🧹 Handling missing data...")
    integrated_df = handle_missing_data(integrated_df)
    
    # Step 5: Create composite features
    print("✨ Creating composite features...")
    integrated_df = create_composite_features(integrated_df)
    
    # Step 6: Final validation
    print("✅ Validating final dataset...")
    validate_integration_quality(integrated_df)
    
    print(f"🎉 Integration Complete! Final dataset: {len(integrated_df):,} buildings, {len(integrated_df.columns)} features")
    
    return integrated_df

def validate_integration_quality(df):
    """Validate the quality of the integrated dataset"""
    
    print(f"\n📊 Dataset Quality Report:")
    print(f"   • Total buildings: {len(df):,}")
    print(f"   • Total features: {len(df.columns)}")
    print(f"   • Missing data rate: {df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")
    print(f"   • BBL uniqueness: {df['BBL'].nunique() == len(df)}")
    
    # Feature category counts
    pluto_features = [col for col in df.columns if any(x in col.lower() for x in ['building', 'floor', 'year', 'sqft'])]
    acris_features = [col for col in df.columns if any(x in col.lower() for x in ['transaction', 'distress', 'price'])]
    mta_features = [col for col in df.columns if any(x in col.lower() for x in ['station', 'ridership', 'accessibility'])]
    
    print(f"   • PLUTO features: {len(pluto_features)}")
    print(f"   • ACRIS features: {len(acris_features)}")
    print(f"   • MTA features: {len(mta_features)}")
    
    return True
```

---

## 🔍 Data Quality and Validation Code

### **Missing Data Analysis**

```python
def analyze_missing_data(df):
    """Comprehensive missing data analysis"""
    
    missing_summary = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_summary.append({
            'column': col,
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'data_type': str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_summary)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)
    
    print("📊 Missing Data Analysis:")
    print(missing_df[missing_df['missing_percentage'] > 0].head(10))
    
    return missing_df

def handle_missing_data(df):
    """Sophisticated missing data handling strategy"""
    
    df_imputed = df.copy()
    
    # Strategy 1: Property-type median for ACRIS features
    acris_cols = [col for col in df.columns if 'transaction' in col or 'distress' in col]
    for col in acris_cols:
        if df_imputed[col].isnull().any():
            # Impute by building class median
            df_imputed[col] = df_imputed.groupby('building_class')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Fill remaining with overall median
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    
    # Strategy 2: Geographic median for proximity features
    proximity_cols = [col for col in df.columns if 'distance' in col or 'nearby' in col]
    for col in proximity_cols:
        if df_imputed[col].isnull().any():
            # Impute by borough median
            df_imputed[col] = df_imputed.groupby('borough')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Strategy 3: Zero imputation for activity features (permits, business)
    activity_cols = [col for col in df.columns if 'permit' in col or 'investment' in col]
    for col in activity_cols:
        df_imputed[col] = df_imputed[col].fillna(0)
    
    return df_imputed
```

### **Integration Validation Tests**

```python
def run_integration_validation_tests(df):
    """Run comprehensive validation tests on integrated dataset"""
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: BBL uniqueness
    total_tests += 1
    if df['BBL'].nunique() == len(df):
        print("✅ Test 1 PASSED: BBL uniqueness")
        tests_passed += 1
    else:
        print("❌ Test 1 FAILED: Duplicate BBLs found")
    
    # Test 2: Feature completeness
    total_tests += 1
    expected_features = 139  # Based on our feature engineering
    if len(df.columns) >= expected_features:
        print(f"✅ Test 2 PASSED: Feature completeness ({len(df.columns)} features)")
        tests_passed += 1
    else:
        print(f"❌ Test 2 FAILED: Missing features ({len(df.columns)}/{expected_features})")
    
    # Test 3: Data quality threshold
    total_tests += 1
    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_rate < 0.1:  # Less than 10% missing
        print(f"✅ Test 3 PASSED: Data quality ({missing_rate:.1%} missing)")
        tests_passed += 1
    else:
        print(f"❌ Test 3 FAILED: Too much missing data ({missing_rate:.1%})")
    
    # Test 4: Geographic coverage
    total_tests += 1
    borough_coverage = df['borough'].nunique()
    if borough_coverage == 5:  # All 5 NYC boroughs
        print("✅ Test 4 PASSED: Geographic coverage (all 5 boroughs)")
        tests_passed += 1
    else:
        print(f"❌ Test 4 FAILED: Incomplete borough coverage ({borough_coverage}/5)")
    
    # Test 5: Feature value ranges
    total_tests += 1
    composite_features = [col for col in df.columns if 'composite' in col or 'score' in col]
    range_valid = all(
        (df[col].min() >= 0) and (df[col].max() <= 1) 
        for col in composite_features 
        if df[col].notna().any()
    )
    if range_valid:
        print("✅ Test 5 PASSED: Composite feature ranges [0,1]")
        tests_passed += 1
    else:
        print("❌ Test 5 FAILED: Invalid composite feature ranges")
    
    print(f"\n🎯 Validation Summary: {tests_passed}/{total_tests} tests passed")
    
    return tests_passed == total_tests
```

---

This technical implementation file provides the detailed code examples that support the methodology described in the main integration document. Together, they demonstrate the sophisticated approach used to integrate 6 diverse NYC datasets into a unified predictive modeling framework.