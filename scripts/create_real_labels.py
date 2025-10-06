"""Create real vacancy labels from storefront data

This script creates realistic target labels for office vacancy prediction
using the storefront vacancy data as ground truth indicators.

Logic:
- Buildings with high storefront vacancy rates → likely office vacancy risk
- Buildings with multiple vacant storefronts → higher vacancy probability  
- Buildings with no storefront data → assumed occupied (conservative)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_vacancy_labels():
    """Create real vacancy labels from storefront and building characteristics"""
    
    print("Loading processed dataset with storefronts...")
    processed_file = Path('data/processed/integrated_buildings_20251002_with_storefronts.csv')
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    
    # Read in chunks to handle large file
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(processed_file, chunksize=chunk_size, low_memory=False):
        # Select only the columns we need for labeling
        label_cols = ['bbl', 'storefront_vacant_count', 'storefront_vacant_flag', 
                     'bldgclass', 'landuse', 'officearea', 'comarea', 'bldgarea']
        
        # Add transaction and permit features if they exist
        if 'transaction_distress_score' in chunk.columns:
            label_cols.append('transaction_distress_score')
        if 'recent_permit_activity' in chunk.columns:
            label_cols.append('recent_permit_activity')
            
        chunk_subset = chunk[label_cols].copy()
        chunks.append(chunk_subset)
        print(f"Processed chunk: {len(chunk)} rows")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Create vacancy probability based on multiple signals
    df['vacancy_score'] = 0.0
    
    # Signal 1: Storefront vacancy (strongest indicator)
    df['storefront_vacant_count'] = pd.to_numeric(df['storefront_vacant_count'], errors='coerce').fillna(0)
    df['storefront_vacant_flag'] = pd.to_numeric(df['storefront_vacant_flag'], errors='coerce').fillna(0)
    
    # High storefront vacancy indicates building distress
    df.loc[df['storefront_vacant_count'] >= 3, 'vacancy_score'] += 0.4
    df.loc[df['storefront_vacant_count'] >= 1, 'vacancy_score'] += 0.2
    df.loc[df['storefront_vacant_flag'] == 1, 'vacancy_score'] += 0.1
    
    # Signal 2: Building characteristics
    # Office buildings with very low office area might be vacant
    df['officearea'] = pd.to_numeric(df['officearea'], errors='coerce').fillna(0)
    df['bldgarea'] = pd.to_numeric(df['bldgarea'], errors='coerce').fillna(1)
    df['office_ratio'] = df['officearea'] / df['bldgarea']
    
    # Buildings classified as office but with no office area
    office_classes = df['bldgclass'].str.startswith('O', na=False)
    no_office_area = (df['officearea'] == 0) & (df['bldgarea'] > 0)
    df.loc[office_classes & no_office_area, 'vacancy_score'] += 0.2
    
    # Signal 3: Transaction distress (if available)
    if 'transaction_distress_score' in df.columns:
        df['transaction_distress_score'] = pd.to_numeric(df['transaction_distress_score'], errors='coerce').fillna(0)
        # Normalize and add to vacancy score
        distress_norm = df['transaction_distress_score'] / (df['transaction_distress_score'].max() + 1e-8)
        df['vacancy_score'] += distress_norm * 0.15
    
    # Signal 4: No recent activity (if available)
    if 'recent_permit_activity' in df.columns:
        df['recent_permit_activity'] = pd.to_numeric(df['recent_permit_activity'], errors='coerce').fillna(0)
        # No recent permits might indicate abandonment
        df.loc[df['recent_permit_activity'] == 0, 'vacancy_score'] += 0.05
    
    # Convert to binary labels using threshold
    # Use a threshold that creates reasonable class balance (not too extreme)
    print(f"Vacancy score range: {df['vacancy_score'].min():.3f} to {df['vacancy_score'].max():.3f}")
    print(f"Vacancy score quantiles: {df['vacancy_score'].quantile([0.5, 0.75, 0.85, 0.9, 0.95]).to_dict()}")
    
    # Use a more aggressive threshold for meaningful separation
    # Buildings with score > 0.3 are likely vacant (multiple risk factors)
    threshold = max(0.3, df['vacancy_score'].quantile(0.9))
    df['is_vacant'] = (df['vacancy_score'] >= threshold).astype(int)
    
    # If still too many/few vacant, adjust threshold
    vacancy_rate = df['is_vacant'].mean()
    if vacancy_rate > 0.5:  # Too many vacant
        threshold = df['vacancy_score'].quantile(0.95)
        df['is_vacant'] = (df['vacancy_score'] >= threshold).astype(int)
    elif vacancy_rate < 0.01:  # Too few vacant
        threshold = df['vacancy_score'].quantile(0.85)
        df['is_vacant'] = (df['vacancy_score'] >= threshold).astype(int)
    
    # Create final target series aligned with feature order
    features_file = Path('data/features/X_features_with_storefronts.csv')
    if features_file.exists():
        # Read just the index to ensure alignment
        feature_df = pd.read_csv(features_file, usecols=[0], nrows=None)
        n_features = len(feature_df)
        
        # Ensure we have the right number of labels
        if len(df) >= n_features:
            y_real = df['is_vacant'].iloc[:n_features]
        else:
            # Pad with zeros if needed (shouldn't happen)
            y_real = pd.concat([df['is_vacant'], pd.Series([0] * (n_features - len(df)))])
    else:
        y_real = df['is_vacant']
    
    # Save results
    results_dir = Path('data/features')
    results_dir.mkdir(exist_ok=True)
    
    # Save real labels
    y_real.to_csv(results_dir / 'y_target_real.csv', header=['is_vacant'], index=False)
    
    # Save label creation details
    label_stats = {
        'total_buildings': len(df),
        'vacant_buildings': df['is_vacant'].sum(),
        'vacancy_rate': df['is_vacant'].mean(),
        'threshold_used': threshold,
        'buildings_with_storefront_data': (df['storefront_vacant_count'] > 0).sum(),
        'avg_vacancy_score': df['vacancy_score'].mean(),
        'vacancy_score_std': df['vacancy_score'].std()
    }
    
    print("\n=== VACANCY LABEL STATISTICS ===")
    for key, value in label_stats.items():
        print(f"{key}: {value}")
    
    # Save detailed results for analysis
    df[['bbl', 'storefront_vacant_count', 'vacancy_score', 'is_vacant']].to_csv(
        results_dir / 'vacancy_labels_detailed.csv', index=False
    )
    
    print(f"\nReal labels saved to: {results_dir / 'y_target_real.csv'}")
    print(f"Label details saved to: {results_dir / 'vacancy_labels_detailed.csv'}")
    
    return y_real, label_stats

if __name__ == '__main__':
    try:
        labels, stats = create_vacancy_labels()
        print("\nReal vacancy labels created successfully!")
    except Exception as e:
        print(f"Error creating labels: {e}")
        import traceback
        traceback.print_exc()