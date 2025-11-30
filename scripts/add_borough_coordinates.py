#!/usr/bin/env python3
"""
Add Borough-Based Geographic Data to Office Buildings
Since BBL matching is complex, we'll add borough information and approximate coordinates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_borough_coordinates():
    """Add borough information and approximate coordinates."""
    
    print("ADDING BOROUGH-BASED GEOGRAPHIC DATA")
    print("=" * 50)
    
    # Load office buildings
    office_df = pd.read_csv('data/processed/office_buildings_clean.csv')
    print(f"Office buildings: {len(office_df):,} records")
    
    # Extract borough from BBL (first digit)
    borough_map = {
        1: 'Manhattan',
        2: 'Bronx', 
        3: 'Brooklyn',
        4: 'Queens',
        5: 'Staten Island'
    }
    
    # Borough centroids (approximate)
    borough_coords = {
        'Manhattan': {'lat': 40.7831, 'lon': -73.9712},
        'Bronx': {'lat': 40.8448, 'lon': -73.8648}, 
        'Brooklyn': {'lat': 40.6782, 'lon': -73.9442},
        'Queens': {'lat': 40.7282, 'lon': -73.7949},
        'Staten Island': {'lat': 40.5795, 'lon': -74.1502}
    }
    
    # Extract borough from BBL
    office_df['borough_code'] = office_df['BBL'].astype(str).str[0].astype(int)
    office_df['borough'] = office_df['borough_code'].map(borough_map)
    
    # Add approximate coordinates (with some random variation for visualization)
    np.random.seed(42)
    
    office_df['latitude'] = office_df['borough'].map(lambda x: borough_coords[x]['lat'] if x in borough_coords else np.nan)
    office_df['longitude'] = office_df['borough'].map(lambda x: borough_coords[x]['lon'] if x in borough_coords else np.nan)
    
    # Add random variation to spread points around borough centroids
    lat_variation = np.random.normal(0, 0.05, len(office_df))  # ~3 mile radius
    lon_variation = np.random.normal(0, 0.05, len(office_df))
    
    office_df['latitude'] += lat_variation
    office_df['longitude'] += lon_variation
    
    # Analysis
    print(f"\nBorough Distribution:")
    borough_counts = office_df['borough'].value_counts()
    for borough, count in borough_counts.items():
        print(f"  {borough}: {count:,} buildings ({count/len(office_df):.1%})")
    
    # Save enhanced dataset
    output_path = 'data/processed/office_buildings_with_coordinates.csv'
    office_df.to_csv(output_path, index=False)
    print(f"\n✅ Enhanced dataset saved: {output_path}")
    
    return office_df

def create_borough_map(df):
    """Create a borough-based map visualization."""
    
    print(f"\nCreating borough map with {len(df):,} buildings...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color map for boroughs
    borough_colors = {
        'Manhattan': '#FF6B6B',
        'Brooklyn': '#4ECDC4', 
        'Queens': '#45B7D1',
        'Bronx': '#96CEB4',
        'Staten Island': '#FFEAA7'
    }
    
    # Plot 1: Buildings by borough
    for borough in df['borough'].unique():
        borough_data = df[df['borough'] == borough]
        ax1.scatter(
            borough_data['longitude'], 
            borough_data['latitude'],
            c=borough_colors.get(borough, 'gray'),
            alpha=0.6, 
            s=20, 
            label=f"{borough} ({len(borough_data):,})",
            edgecolors='white',
            linewidth=0.3
        )
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude') 
    ax1.set_title('NYC Office Buildings by Borough\n(Approximate Locations)', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Risk by borough
    if 'target_high_vacancy_risk' in df.columns:
        risk_colors = df['target_high_vacancy_risk'].map({0: 'green', 1: 'red'})
        ax2.scatter(df['longitude'], df['latitude'], 
                   c=risk_colors, alpha=0.6, s=25, edgecolors='black', linewidth=0.2)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Vacancy Risk Distribution\n🟢 Low Risk  🔴 High Risk', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Risk statistics by borough
        risk_by_borough = df.groupby('borough')['target_high_vacancy_risk'].agg(['mean', 'count'])
        print(f"\nRisk by Borough:")
        for borough, stats in risk_by_borough.iterrows():
            print(f"  {borough}: {stats['mean']:.1%} high risk ({int(stats['count'])} buildings)")
    
    plt.tight_layout()
    plt.savefig('figures/nyc_office_buildings_borough_map.png', dpi=300, bbox_inches='tight')
    print("Borough map saved: figures/nyc_office_buildings_borough_map.png")
    plt.show()

def suggest_exact_coordinates():
    """Suggest methods for getting exact coordinates."""
    
    print(f"\n{'='*60}")
    print("OPTIONS FOR EXACT BUILDING COORDINATES")
    print(f"{'='*60}")
    
    print("📍 Current Solution: Borough-level mapping with approximate coordinates")
    print("   ✅ Works immediately for dashboard geographic visualization")
    print("   ✅ Shows risk patterns by borough")
    print("   ⚠️  Not building-specific locations")
    
    print("\n🔍 For Exact Coordinates, Consider:")
    
    print("\n1. 📡 NYC Geoclient API")
    print("   - Input: Address or BBL → Output: Exact lat/lon")
    print("   - Free for NYC data")
    print("   - API: https://developer.cityofnewyork.us/api/geoclient-api")
    
    print("\n2. 📊 Alternative NYC Open Data")
    print("   - Property Address Directory (PAD)")
    print("   - Building Footprints dataset") 
    print("   - MapPLUTO (different version)")
    
    print("\n3. 🗺️ Google/OpenStreetMap Geocoding")
    print("   - If we have addresses in other data sources")
    print("   - Bulk geocoding services")
    
    print("\n4. 🔄 BBL Format Investigation")
    print("   - Our BBLs: 11 digits (e.g., 41174000100)")
    print("   - PLUTO BBLs: 10 digits (e.g., 4064210038)")
    print("   - May need BBL conversion logic")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION: Use current borough solution for dashboard demo,")
    print("then investigate exact coordinates for production version.")
    print(f"{'='*60}")

def main():
    """Main execution."""
    
    # Add borough-based coordinates
    df = add_borough_coordinates()
    
    # Create visualization
    create_borough_map(df)
    
    # Suggest exact coordinate options
    suggest_exact_coordinates()
    
    print(f"\n🗺️ GEOGRAPHIC DATA READY!")
    print(f"✅ Dashboard can now show borough-based risk mapping")
    print(f"📁 File: data/processed/office_buildings_with_coordinates.csv")

if __name__ == "__main__":
    main()