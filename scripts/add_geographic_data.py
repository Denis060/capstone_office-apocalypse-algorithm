#!/usr/bin/env python3
"""
Add Geographic Coordinates to Office Buildings Dataset
Uses NYC PLUTO data to add lat/lon coordinates based on BBL numbers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_pluto_coordinates():
    """Load coordinate data from PLUTO dataset."""
    print("Loading PLUTO coordinate data...")
    
    # Load only the columns we need to save memory
    pluto_cols = ['bbl', 'latitude', 'longitude', 'xcoord', 'ycoord', 'borough']
    
    try:
        pluto = pd.read_csv('data/raw/pluto_25v2_1.csv', usecols=pluto_cols)
        print(f"Loaded PLUTO data: {len(pluto):,} records")
        
        # Clean up coordinate data
        pluto = pluto[
            (pluto['latitude'].notna()) & 
            (pluto['longitude'].notna()) & 
            (pluto['latitude'] != 0) & 
            (pluto['longitude'] != 0)
        ]
        
        print(f"Valid coordinates: {len(pluto):,} records")
        return pluto
        
    except Exception as e:
        print(f"Error loading PLUTO data: {e}")
        return None

def merge_coordinates():
    """Merge coordinates with office buildings data."""
    
    # Load office buildings
    print("Loading office buildings data...")
    office_df = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Office buildings: {len(office_df):,} records")
    
    # Load PLUTO coordinates  
    pluto_df = load_pluto_coordinates()
    if pluto_df is None:
        return None
    
    # Handle BBL format differences
    # Office BBLs are 11 digits, PLUTO BBLs are 10 digits (missing leading zero in some cases)
    office_df['BBL_int'] = pd.to_numeric(office_df['BBL'], errors='coerce')
    pluto_df['bbl_int'] = pd.to_numeric(pluto_df['bbl'], errors='coerce')
    
    print(f"\nSample office BBLs: {office_df['BBL'].head().tolist()}")
    print(f"Sample PLUTO BBLs: {pluto_df['bbl'].head().tolist()}")
    print(f"Office BBL as int: {office_df['BBL_int'].head().tolist()}")
    print(f"PLUTO BBL as int: {pluto_df['bbl_int'].head().tolist()}")
    
    # Try multiple matching strategies
    merged_df = None
    match_strategies = [
        ("Direct integer match", 'BBL_int', 'bbl_int'),
        ("String match", 'BBL', 'bbl'),
    ]
    
    for strategy_name, left_col, right_col in match_strategies:
        print(f"\nTrying {strategy_name}...")
        
        # Prepare columns for matching
        if left_col == 'BBL':
            office_df['BBL_clean'] = office_df['BBL'].astype(str).str.strip()
            left_match_col = 'BBL_clean'
        else:
            left_match_col = left_col
            
        if right_col == 'bbl':
            pluto_df['bbl_clean'] = pluto_df['bbl'].astype(str).str.strip().str.replace('.0', '')
            right_match_col = 'bbl_clean'
        else:
            right_match_col = right_col
        
        # Attempt merge
        temp_merged = office_df.merge(
            pluto_df[['bbl', right_match_col, 'latitude', 'longitude', 'borough']], 
            left_on=left_match_col, 
            right_on=right_match_col, 
            how='left'
        )
        
        with_coords = temp_merged[temp_merged['latitude'].notna()].shape[0]
        match_rate = with_coords / len(temp_merged)
        
        print(f"  Match rate: {match_rate:.1%} ({with_coords:,} buildings)")
        
        if merged_df is None or with_coords > merged_df[merged_df['latitude'].notna()].shape[0]:
            merged_df = temp_merged
            best_strategy = strategy_name
            best_match_rate = match_rate
            best_with_coords = with_coords
    
    if merged_df is not None and best_with_coords > 0:
        print(f"\n✅ Best strategy: {best_strategy}")
        print(f"   Final match rate: {best_match_rate:.1%} ({best_with_coords:,} buildings)")
        
        # Clean up merged dataset
        cols_to_drop = [col for col in merged_df.columns if col in ['bbl', 'BBL_int', 'bbl_int', 'BBL_clean', 'bbl_clean']]
        merged_df = merged_df.drop(columns=cols_to_drop)
        
        # Add borough mapping if missing
        if 'borough' not in merged_df.columns or merged_df['borough'].isna().all():
            borough_map = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'}
            merged_df['borough'] = merged_df['BBL'].astype(str).str[0].map(lambda x: borough_map.get(int(x), 'Unknown'))
        
        # Save enhanced dataset
        output_path = 'data/processed/office_buildings_with_coordinates.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Enhanced dataset saved: {output_path}")
        
        return merged_df, best_with_coords, best_match_rate
    
    else:
        print("❌ No coordinate matches found with any strategy")
        return None, 0, 0

def analyze_geographic_coverage(df, with_coords, match_rate):
    """Analyze geographic coverage of the enhanced dataset."""
    
    print(f"\n{'='*60}")
    print("GEOGRAPHIC ANALYSIS")
    print(f"{'='*60}")
    
    # Overall stats
    print(f"Buildings with coordinates: {with_coords:,} ({match_rate:.1%})")
    
    # Borough breakdown
    if 'borough' in df.columns:
        borough_stats = df[df['latitude'].notna()]['borough'].value_counts()
        print(f"\nBuildings by Borough:")
        for borough, count in borough_stats.items():
            print(f"  {borough}: {count:,}")
    
    # Coordinate ranges
    coords_df = df[df['latitude'].notna()]
    print(f"\nCoordinate Ranges:")
    print(f"  Latitude: {coords_df['latitude'].min():.4f} to {coords_df['latitude'].max():.4f}")
    print(f"  Longitude: {coords_df['longitude'].min():.4f} to {coords_df['longitude'].max():.4f}")
    
    # Risk distribution by geography
    if 'target_high_vacancy_risk' in df.columns and with_coords > 0:
        coords_df = df[df['latitude'].notna()].copy()
        risk_by_borough = coords_df.groupby('borough')['target_high_vacancy_risk'].agg(['mean', 'count'])
        
        print(f"\nRisk by Borough:")
        print(risk_by_borough.round(3))

def create_sample_map():
    """Create a sample geographic visualization."""
    
    try:
        # Load the enhanced dataset
        df = pd.read_csv('data/processed/office_buildings_with_coordinates.csv')
        
        if 'latitude' not in df.columns:
            print("No coordinate data found in enhanced dataset")
            return
        
        coords_df = df[
            (df['latitude'].notna()) & 
            (df['longitude'].notna()) &
            (df['latitude'] != 0) & 
            (df['longitude'] != 0)
        ].copy()
        
        if len(coords_df) == 0:
            print("No valid coordinates found")
            return
        
        print(f"\nCreating sample map with {len(coords_df):,} buildings...")
        
        # Create risk map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All buildings
        ax1.scatter(coords_df['longitude'], coords_df['latitude'], 
                   alpha=0.6, s=20, c='lightblue', edgecolors='navy', linewidth=0.5)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'NYC Office Buildings\n{len(coords_df):,} buildings', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Risk-colored map
        if 'target_high_vacancy_risk' in coords_df.columns:
            risk_colors = coords_df['target_high_vacancy_risk'].map({0: 'green', 1: 'red'})
            ax2.scatter(coords_df['longitude'], coords_df['latitude'], 
                       c=risk_colors, alpha=0.7, s=25, edgecolors='black', linewidth=0.3)
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('Vacancy Risk by Location\n🟢 Low Risk  🔴 High Risk', fontweight='bold')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/nyc_office_buildings_map.png', dpi=300, bbox_inches='tight')
        print("Sample map saved: figures/nyc_office_buildings_map.png")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating sample map: {e}")

def main():
    """Main execution function."""
    
    print("NYC OFFICE BUILDINGS - ADDING GEOGRAPHIC COORDINATES")
    print("=" * 60)
    
    # Merge coordinates
    result = merge_coordinates()
    
    if result is None:
        print("❌ Failed to add coordinates")
        return
    
    df, with_coords, match_rate = result
    
    if with_coords > 0:
        # Analyze coverage
        analyze_geographic_coverage(df, with_coords, match_rate)
        
        # Create sample map
        create_sample_map()
        
        print(f"\n✅ SUCCESS!")
        print(f"Enhanced dataset ready for dashboard mapping!")
        print(f"File: data/processed/office_buildings_with_coordinates.csv")
        
    else:
        print("❌ No geographic coordinates could be added")
        print("Consider alternative data sources or BBL format investigation")

if __name__ == "__main__":
    main()