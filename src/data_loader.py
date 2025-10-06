"""
Data Loading and Integration Module for Office Apocalypse Algorithm

This module implements the core data integration strategy for predicting NYC office vacancy.
The approach uses a BBL (Borough-Block-Lot) centric methodology to merge multiple datasets
around the commercial office building as the central unit of analysis.

INTEGRATION STRATEGY OVERVIEW:
============================
1. Start with PLUTO dataset as the "universe" of all NYC buildings
2. Join ACRIS transaction data using BBL to add financial distress signals
3. Add geospatial MTA ridership data for proximity-based demand indicators
4. Join business registry data for economic activity indicators
5. Merge tax assessment data for valuation and financial stress metrics
6. Create integrated dataset ready for feature engineering

KEY ASSUMPTIONS:
===============
- BBL serves as reliable primary key across datasets
- Missing data in joins indicates "no activity" rather than data errors
- Geospatial operations require coordinate data (to be added in future iterations)
- Public datasets are sufficiently accurate for predictive modeling

DATA QUALITY CONSIDERATIONS:
===========================
- All joins use left joins to preserve PLUTO universe
- Missing values filled with appropriate defaults (0 for counts, None for coordinates)
- Data types explicitly defined to prevent pandas inference issues
- Error handling prevents pipeline failures from single dataset issues
"""

import pandas as pd
import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import numpy as np


def load_pluto_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load PLUTO (Primary Land Use Tax Lot Output) dataset.

    WHY PLUTO FIRST?
    ================
    PLUTO serves as the "universe" of all NYC buildings and tax lots. Every merge
    starts from this dataset to ensure we have complete coverage of NYC's building stock.
    The BBL (Borough-Block-Lot) identifier is our primary key for joining all other datasets.

    DATA CHARACTERISTICS:
    ====================
    - ~857K rows representing all NYC tax lots
    - Contains building physical attributes (age, size, zoning)
    - Includes valuation data and land use classifications
    - Critical for identifying office buildings vs other property types

    Args:
        data_dir: Directory containing the raw data files (default: "data/raw")

    Returns:
        DataFrame with PLUTO data, explicitly typed for consistency

    Raises:
        FileNotFoundError: If PLUTO file is not found in specified directory
    """
    file_path = Path(data_dir) / "pluto_25v2_1.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"PLUTO data file not found at {file_path}")

    # EXPLICIT DATA TYPES: Prevent pandas from inferring wrong types
    # This is crucial for large datasets where inference can be slow and inaccurate
    dtypes = {
        'bbl': str,           # Primary key - must be string to preserve leading zeros
        'borough': str,       # Borough name/code
        'block': 'Int64',     # Tax block number (nullable integer)
        'lot': 'Int64',       # Tax lot number (nullable integer)
        'zipcode': 'Int64',   # ZIP code (nullable integer)
        'address': str,       # Street address
        'bldgclass': str,     # Building classification code
        'landuse': str,       # Land use category
        'numfloors': float,   # Number of floors (float for partial floors)
        'yearbuilt': 'Int64', # Construction year (nullable integer)
        'lotarea': 'Int64',   # Lot area in square feet
        'bldgarea': 'Int64',  # Building area in square feet
        'comarea': 'Int64',   # Commercial area in square feet
        'officearea': 'Int64',# Office area in square feet (TARGET VARIABLE PROXY)
        'assessland': float,  # Land assessed value
        'assesstot': float    # Total assessed value
    }

    df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    print(f"Loaded PLUTO data: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_acris_data(data_dir: str = "data/raw", sample_fraction: float = 0.1) -> pd.DataFrame:
    """
    Load ACRIS (Automated City Register Information System) dataset.

    WHY ACRIS IS IMPORTANT:
    ======================
    ACRIS contains the complete history of property transactions in NYC, including:
    - Sales prices and dates
    - Mortgage recordings and amounts
    - Foreclosure filings and auctions
    - Property transfers and assignments

    These financial transactions provide critical signals of building distress:
    - Frequent short sales may indicate financial trouble
    - Foreclosures signal imminent vacancy risk
    - No recent transactions may indicate lack of investment interest

    MEMORY OPTIMIZATION:
    ===================
    ACRIS dataset contains ~22M records. For development and EDA, we sample
    a fraction of the data to reduce memory usage while maintaining statistical
    properties. In production, consider aggregating by BBL instead.

    Args:
        data_dir: Directory containing the raw data files
        sample_fraction: Fraction of data to sample (0.1 = 10%)

    Returns:
        DataFrame with ACRIS transaction data (sampled)

    Note: In practice, ACRIS data needs preprocessing to extract BBL identifiers
    and transaction types. This is a simplified loader for the framework.
    """
    file_path = Path(data_dir) / "ACRIS_-_Real_Property_Legals_20250915.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"ACRIS data file not found at {file_path}")

    # MEMORY OPTIMIZATION: Sample a fraction of the data for development
    # This reduces 22M rows to ~2.2M rows, making it manageable
    import numpy as np
    df = pd.read_csv(file_path, low_memory=False, skiprows=lambda i: i > 0 and np.random.random() > sample_fraction)

    # Create BBL (Borough-Block-Lot) identifier from component fields
    # BBL format: BBLLLLLLLL (1-digit borough + 5-digit block + 4-digit lot, zero-padded)
    try:
        df['bbl'] = df['BOROUGH'].astype(str) + \
                    df['BLOCK'].astype(str).str.zfill(5) + \
                    df['LOT'].astype(str).str.zfill(4)
    except (KeyError, MemoryError) as e:
        print(f"Warning: Could not create BBL from ACRIS data: {e}")
        df['bbl'] = None

    print(f"Loaded ACRIS data (sampled {sample_fraction*100:.0f}%): {len(df)} rows, {len(df.columns)} columns")
    return df


def load_mta_ridership(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load MTA Subway Hourly Ridership dataset with memory optimization.

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        DataFrame with MTA ridership data (sampled for memory efficiency)
    """
    file_path = Path(data_dir) / "MTA_Subway_Hourly_Ridership__2020-2024.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"MTA ridership file not found at {file_path}")

    # Sample aggressively to avoid memory issues
    df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.random() > 0.01, low_memory=False)  # 1% sample
    print(f"Loaded MTA ridership (1% sample): {len(df)} rows, {len(df.columns)} columns")
    return df


def load_business_registry(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load NYC Business Registry dataset.

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        DataFrame with business registry data
    """
    file_path = Path(data_dir) / "business_registry.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Business registry file not found at {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded business registry: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_dob_permits(data_dir: str = "data/raw", sample_size: int = 50000) -> pd.DataFrame:
    """
    Load DOB Permit Issuance dataset.

    Args:
        data_dir: Directory containing the raw data files
        sample_size: Number of rows to sample for development

    Returns:
        DataFrame with DOB permit data
    """
    file_path = Path(data_dir) / "DOB_Permit_Issuance_20250915.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"DOB permits file not found at {file_path}")

    df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
    print(f"Loaded DOB permits (sampled): {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_acris_to_pluto(pluto_df: pd.DataFrame, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Merge ACRIS property transaction data with PLUTO using memory-efficient aggregation.

    ENHANCED MERGING STRATEGY:
    =========================
    - Aggregate ACRIS data by BBL before merging to reduce memory usage
    - Create transaction frequency and recency features
    - Use left join to preserve all PLUTO buildings
    - Handle missing transaction data as "no recent activity"

    TRANSACTION FEATURES CREATED:
    ============================
    - transaction_count_last_3y: Number of transactions in last 3 years
    - transaction_count_last_1y: Number of transactions in last year
    - has_foreclosure_activity: Boolean for foreclosure-related transactions
    - days_since_last_transaction: Recency of last transaction
    - transaction_distress_score: Composite score for financial distress

    Args:
        pluto_df: PLUTO DataFrame (base dataset)
        data_dir: Directory containing raw data files

    Returns:
        DataFrame with ACRIS transaction features added
    """
    try:
        print("Loading ACRIS data for aggregation...")
        acris_df = load_acris_data(data_dir, sample_fraction=0.5)  # Use 50% sample for development

        # Aggregate by BBL to create transaction features
        print("Aggregating ACRIS data by BBL...")

        # Basic transaction counts by BBL
        transaction_counts = acris_df.groupby('bbl').size().reset_index(name='total_transaction_count')

        # For now, create simplified features (would expand with date parsing)
        acris_features = transaction_counts.copy()

        # Add placeholder features for distress indicators
        acris_features['transaction_count_last_3y'] = acris_features['total_transaction_count']  # Placeholder
        acris_features['transaction_count_last_1y'] = (acris_features['total_transaction_count'] * 0.3).astype(int)  # Placeholder
        acris_features['has_foreclosure_activity'] = 0  # Placeholder - would need transaction type analysis
        acris_features['days_since_last_transaction'] = 365  # Placeholder - would need date parsing
        acris_features['transaction_distress_score'] = acris_features['total_transaction_count'] * 0.1  # Placeholder

        print(f"ACRIS features created for {len(acris_features)} unique BBLs")

        # LEFT JOIN: Keep all PLUTO buildings, add transaction data where available
        merged_df = pluto_df.merge(acris_features, on='bbl', how='left', suffixes=('', '_acris'))

        # Fill missing transaction data with 0 (no activity)
        transaction_cols = ['total_transaction_count', 'transaction_count_last_3y',
                          'transaction_count_last_1y', 'has_foreclosure_activity',
                          'transaction_distress_score']
        merged_df[transaction_cols] = merged_df[transaction_cols].fillna(0)
        merged_df['days_since_last_transaction'] = merged_df['days_since_last_transaction'].fillna(9999)  # Large number = no recent activity

        print(f"Merged ACRIS features: {len(merged_df)} rows after merge")
        print(f"Buildings with transaction history: {(merged_df['total_transaction_count'] > 0).sum()}")

        return merged_df

    except FileNotFoundError:
        print("ACRIS data not found, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['total_transaction_count'] = 0
        pluto_df['transaction_count_last_3y'] = 0
        pluto_df['transaction_count_last_1y'] = 0
        pluto_df['has_foreclosure_activity'] = 0
        pluto_df['days_since_last_transaction'] = 9999
        pluto_df['transaction_distress_score'] = 0
        return pluto_df
    except Exception as e:
        print(f"Error in ACRIS merge: {e}, returning PLUTO unchanged")
        pluto_df = pluto_df.copy()
        pluto_df['total_transaction_count'] = 0
        pluto_df['transaction_count_last_3y'] = 0
        pluto_df['transaction_count_last_1y'] = 0
        pluto_df['has_foreclosure_activity'] = 0
        pluto_df['days_since_last_transaction'] = 9999
        pluto_df['transaction_distress_score'] = 0
        return pluto_df


def merge_acris_to_pluto_chunked(pluto_df: pd.DataFrame, data_dir: str = "data/raw", chunk_size: int = 500000) -> pd.DataFrame:
    """
    Merge ACRIS property transaction data with PLUTO using chunked processing for memory efficiency.

    MEMORY-EFFICIENT CHUNKED APPROACH:
    =================================
    - Process ACRIS data in chunks instead of loading all at once
    - Aggregate transaction features incrementally per chunk
    - Combine results across all chunks
    - Reduces peak memory usage significantly

    Args:
        pluto_df: PLUTO DataFrame (base dataset)
        data_dir: Directory containing raw data files
        chunk_size: Number of rows to process per chunk

    Returns:
        DataFrame with ACRIS transaction features added
    """
    try:
        file_path = Path(data_dir) / "ACRIS_-_Real_Property_Legals_20250915.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"ACRIS data file not found at {file_path}")

        print(f"Processing ACRIS data in chunks of {chunk_size} rows...")

        # Initialize aggregation dictionary
        bbl_aggregations = {}

        # Process file in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
            print(f"Processing chunk {chunk_num + 1}...")

            # Create BBL for this chunk
            try:
                chunk['bbl'] = chunk['BOROUGH'].astype(str) + \
                              chunk['BLOCK'].astype(str).str.zfill(5) + \
                              chunk['LOT'].astype(str).str.zfill(4)
            except (KeyError, AttributeError) as e:
                print(f"Warning: Could not create BBL in chunk {chunk_num + 1}: {e}")
                continue

            # Aggregate transaction counts by BBL for this chunk
            chunk_agg = chunk.groupby('bbl').size().reset_index(name='chunk_count')

            # Accumulate results across chunks
            for _, row in chunk_agg.iterrows():
                bbl = row['bbl']
                count = row['chunk_count']

                if bbl in bbl_aggregations:
                    bbl_aggregations[bbl] += count
                else:
                    bbl_aggregations[bbl] = count

            print(f"  Chunk {chunk_num + 1}: {len(chunk)} rows, {len(chunk_agg)} unique BBLs")

        # Convert aggregations to DataFrame
        acris_features = pd.DataFrame({
            'bbl': list(bbl_aggregations.keys()),
            'total_transaction_count': list(bbl_aggregations.values())
        })

        print(f"Total unique BBLs with transactions: {len(acris_features)}")

        # Create derived features
        acris_features['transaction_count_last_3y'] = acris_features['total_transaction_count']  # Placeholder
        acris_features['transaction_count_last_1y'] = (acris_features['total_transaction_count'] * 0.3).astype(int)  # Placeholder
        acris_features['has_foreclosure_activity'] = 0  # Placeholder - would need transaction type analysis
        acris_features['days_since_last_transaction'] = 365  # Placeholder - would need date parsing
        acris_features['transaction_distress_score'] = acris_features['total_transaction_count'] * 0.1  # Placeholder

        print(f"ACRIS features created for {len(acris_features)} unique BBLs")

        # LEFT JOIN: Keep all PLUTO buildings, add transaction data where available
        merged_df = pluto_df.merge(acris_features, on='bbl', how='left', suffixes=('', '_acris'))

        # Fill missing transaction data with 0 (no activity)
        transaction_cols = ['total_transaction_count', 'transaction_count_last_3y',
                          'transaction_count_last_1y', 'has_foreclosure_activity',
                          'transaction_distress_score']
        merged_df[transaction_cols] = merged_df[transaction_cols].fillna(0)
        merged_df['days_since_last_transaction'] = merged_df['days_since_last_transaction'].fillna(9999)  # Large number = no recent activity

        buildings_with_transactions = (merged_df['total_transaction_count'] > 0).sum()
        print(f"Merged ACRIS features: {len(merged_df)} rows after merge")
        print(f"Buildings with transaction history: {buildings_with_transactions}")

        return merged_df

    except FileNotFoundError:
        print("ACRIS data not found, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['total_transaction_count'] = 0
        pluto_df['transaction_count_last_3y'] = 0
        pluto_df['transaction_count_last_1y'] = 0
        pluto_df['has_foreclosure_activity'] = 0
        pluto_df['days_since_last_transaction'] = 9999
        pluto_df['transaction_distress_score'] = 0
        return pluto_df
    except Exception as e:
        print(f"Error in chunked ACRIS merge: {e}, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['total_transaction_count'] = 0
        pluto_df['transaction_count_last_3y'] = 0
        pluto_df['transaction_count_last_1y'] = 0
        pluto_df['has_foreclosure_activity'] = 0
        pluto_df['days_since_last_transaction'] = 9999
        pluto_df['transaction_distress_score'] = 0
        return pluto_df


def merge_business_to_pluto(pluto_df: pd.DataFrame, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Merge business registry data with PLUTO using ZIP code proximity.

    WHY BUSINESS DATA MATTERS:
    =========================
    The presence and density of active businesses around a building indicates:
    - Economic vitality of the neighborhood
    - Demand for commercial/office space
    - Foot traffic and activity levels
    - Risk of contagion (if nearby businesses are failing)

    MERGING APPROACH:
    ================
    - Aggregate business counts by ZIP code (coarser than building-level)
    - Left join on ZIP code to add business density features
    - Fill missing values with 0 (no businesses in area)

    LIMITATIONS:
    ===========
    ZIP-level aggregation loses granularity but ensures coverage.
    Future improvement: Use geospatial joins for more precise proximity.

    Args:
        pluto_df: PLUTO DataFrame
        data_dir: Directory containing raw data files

    Returns:
        DataFrame with business density features added
    """
    try:
        business_df = load_business_registry(data_dir)
        # Aggregate business counts by ZIP code
        business_agg = business_df.groupby('ZIP Code').size().reset_index(name='business_count_zip')
        
        # Ensure ZIP codes are strings for consistent merging
        business_agg['ZIP Code'] = business_agg['ZIP Code'].astype(str)
        pluto_df_copy = pluto_df.copy()
        pluto_df_copy['zipcode'] = pluto_df_copy['zipcode'].astype(str)

        merged_df = pluto_df_copy.merge(business_agg, left_on='zipcode', right_on='ZIP Code', how='left')
        merged_df['business_count_zip'] = merged_df['business_count_zip'].fillna(0)
        print(f"Merged business data: {len(merged_df)} rows after merge")
        return merged_df
    except FileNotFoundError:
        print("Business registry data not found, returning PLUTO unchanged")
        return pluto_df


def ingest_vacant_storefronts(processed_df: pd.DataFrame, raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Ingest the Vacant Storefronts dataset, aggregate vacancy counts by BBL,
    and merge the results into the processed integrated dataframe.

    Strategy:
    - Use `BBL` or `Borough Block Lot` from the storefronts CSV as the merge key.
    - Aggregate reported vacant flags per BBL (count and binary flag).
    - Read in chunks to handle large file sizes.

    Args:
        processed_df: The already processed/integrated PLUTO-based dataframe
        raw_dir: Directory containing raw data files

    Returns:
        DataFrame with new storefront vacancy features merged
    """
    file_path = Path(raw_dir) / "Storefronts_Reported_Vacant_or_Not_20250915.csv"

    if not file_path.exists():
        print(f"Vacant storefronts file not found at {file_path}. Skipping storefront merge.")
        processed_df['storefront_vacant_count'] = 0
        processed_df['storefront_vacant_flag'] = 0
        return processed_df

    print(f"Ingesting storefronts from {file_path} in chunks...")

    agg_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=200_000, dtype=str, low_memory=True):
        # Normalize BBL column name
        if 'BBL' in chunk.columns:
            chunk['bbl'] = chunk['BBL']
        elif 'Borough Block Lot' in chunk.columns:
            chunk['bbl'] = chunk['Borough Block Lot']
        else:
            # If no BBL, try to locate a column containing 'block' and 'lot'
            candidates = [c for c in chunk.columns if 'block' in c.lower() and 'lot' in c.lower()]
            if candidates:
                chunk['bbl'] = chunk[candidates[0]]
            else:
                chunk['bbl'] = None

        chunk['bbl'] = chunk['bbl'].astype(str).str.strip()

        # Determine vacancy indicator column (prefer explicit columns)
        vac_cols = [c for c in chunk.columns if 'vacant' in c.lower()]
        if vac_cols:
            vac_col = vac_cols[0]
            chunk['is_vacant_reported'] = chunk[vac_col].str.upper().fillna('N').isin(['YES', 'Y', 'TRUE', '1']).astype(int)
        else:
            # Fallback: if Primary Business Activity is blank or 'NO BUSINESS ACTIVITY IDENTIFIED'
            if 'Primary Business Activity' in chunk.columns:
                chunk['is_vacant_reported'] = chunk['Primary Business Activity'].isna().astype(int)
            else:
                chunk['is_vacant_reported'] = 0

        # Aggregate by BBL within this chunk
        grp = chunk.groupby('bbl', dropna=True)['is_vacant_reported'].agg(['sum', 'max']).reset_index()
        grp = grp.rename(columns={'sum': 'storefront_vacant_count_chunk', 'max': 'storefront_vacant_flag_chunk'})
        agg_chunks.append(grp)

    if not agg_chunks:
        print("No storefront data parsed; adding zero columns.")
        processed_df['storefront_vacant_count'] = 0
        processed_df['storefront_vacant_flag'] = 0
        return processed_df

    combined = pd.concat(agg_chunks, ignore_index=True)
    combined_agg = combined.groupby('bbl', dropna=True).agg({
        'storefront_vacant_count_chunk': 'sum',
        'storefront_vacant_flag_chunk': 'max'
    }).reset_index()
    combined_agg = combined_agg.rename(columns={
        'storefront_vacant_count_chunk': 'storefront_vacant_count',
        'storefront_vacant_flag_chunk': 'storefront_vacant_flag'
    })

    # Ensure processed_df has 'bbl' column
    if 'bbl' not in processed_df.columns:
        if 'BBL' in processed_df.columns:
            processed_df['bbl'] = processed_df['BBL'].astype(str)
        else:
            processed_df['bbl'] = processed_df.get('bbl', None)

    # Merge and fill defaults
    processed_df = processed_df.merge(combined_agg, how='left', on='bbl')
    processed_df['storefront_vacant_count'] = processed_df['storefront_vacant_count'].fillna(0).astype(int)
    processed_df['storefront_vacant_flag'] = processed_df['storefront_vacant_flag'].fillna(0).astype(int)

    print(f"Merged storefront vacancy for {combined_agg.shape[0]} BBLs")
    return processed_df


def merge_mta_ridership_geospatial(pluto_df: pd.DataFrame, data_dir: str = "data/raw", radius_meters: int = 500) -> pd.DataFrame:
    """
    Merge MTA ridership data with PLUTO using geospatial proximity.

    WHY MTA DATA IS CRITICAL:
    ========================
    Subway ridership patterns indicate:
    - Commuter demand for office space in the area
    - Economic activity and foot traffic levels
    - Accessibility and transportation convenience
    - Risk of reduced demand if ridership declines

    GEOSPATIAL CHALLENGE:
    ====================
    This requires:
    1. Geocoding PLUTO addresses to latitude/longitude
    2. Having MTA station coordinates and ridership data
    3. Calculating distances between buildings and stations
    4. Aggregating ridership within specified radius

    CURRENT STATUS:
    ==============
    This is a PLACEHOLDER implementation. Real geospatial integration requires:
    - Coordinate data for both buildings and subway stations
    - Spatial indexing for efficient proximity calculations
    - Temporal aggregation of ridership data

    Args:
        pluto_df: PLUTO DataFrame (must have 'latitude' and 'longitude' columns)
        data_dir: Directory containing raw data files
        radius_meters: Search radius around each building

    Returns:
        DataFrame with MTA ridership features added
    """
    try:
        mta_df = load_mta_ridership(data_dir)

        # Aggregate ridership by station (simplified - would need date parsing for full temporal analysis)
        station_ridership = mta_df.groupby(['station_complex', 'latitude', 'longitude']).agg({
            'ridership': 'sum'  # Total ridership for the period
        }).reset_index()

        print(f"Aggregated MTA data: {len(station_ridership)} unique stations")

        # For buildings with coordinates, calculate proximity to stations
        pluto_with_coords = pluto_df.dropna(subset=['latitude', 'longitude']).copy()

        if len(pluto_with_coords) > 0:
            print(f"Calculating proximity for {len(pluto_with_coords)} buildings with coordinates...")

            # Simple proximity calculation (placeholder - would use proper geospatial libraries)
            # For now, create features based on borough proximity to high-ridership areas
            pluto_df = pluto_df.copy()

            # Manhattan gets higher ridership scores
            pluto_df['nearby_stations_count'] = pluto_df['borough'].map({
                'MN': 5, 'BX': 2, 'BK': 3, 'QN': 2, 'SI': 1
            }).fillna(1)

            pluto_df['avg_daily_ridership_500m'] = pluto_df['borough'].map({
                'MN': 15000, 'BX': 3000, 'BK': 5000, 'QN': 4000, 'SI': 1000
            }).fillna(1000)

            pluto_df['total_weekly_ridership_nearby'] = pluto_df['avg_daily_ridership_500m'] * 7
            pluto_df['high_ridership_area'] = (pluto_df['avg_daily_ridership_500m'] > 10000).astype(int)

        else:
            print("No coordinate data available, using borough-based estimates")
            pluto_df = pluto_df.copy()
            pluto_df['nearby_stations_count'] = 1
            pluto_df['avg_daily_ridership_500m'] = 1000.0
            pluto_df['total_weekly_ridership_nearby'] = 7000.0
            pluto_df['high_ridership_area'] = 0

        print(f"MTA ridership features added to {len(pluto_df)} buildings")
        return pluto_df

    except FileNotFoundError:
        print("MTA ridership data not found, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['nearby_stations_count'] = 0
        pluto_df['avg_daily_ridership_500m'] = 0.0
        pluto_df['total_weekly_ridership_nearby'] = 0.0
        pluto_df['high_ridership_area'] = 0
        return pluto_df
    except Exception as e:
        print(f"Error in MTA merge: {e}, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['nearby_stations_count'] = 0
        pluto_df['avg_daily_ridership_500m'] = 0.0
        pluto_df['total_weekly_ridership_nearby'] = 0.0
        pluto_df['high_ridership_area'] = 0
        return pluto_df


def geocode_pluto_addresses(pluto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude/longitude coordinates to PLUTO data using address geocoding.

    This is needed for geospatial joins with MTA and other proximity-based features.

    Args:
        pluto_df: PLUTO DataFrame with Address column

    Returns:
        DataFrame with latitude and longitude columns added
    """
    # Placeholder for geocoding implementation
    # Would use geocoding service (Google Maps API, NYC Geoclient, etc.)
    print("Geocoding: placeholder - implement with geocoding service")
    pluto_df = pluto_df.copy()
    pluto_df['latitude'] = None
    pluto_df['longitude'] = None
    return pluto_df


def apply_missing_value_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-appropriate missing value imputation strategy.

    MISSING VALUE STRATEGY:
    =======================
    1. BUILDING PHYSICAL ATTRIBUTES: Use median/mean for area measurements
    2. AGE FEATURES: Use median building age for missing yearbuilt
    3. FINANCIAL FEATURES: Use 0 for missing transaction/permit activity (no activity)
    4. COORDINATES: Keep as NaN (will be handled in geospatial features)
    5. CATEGORICAL FEATURES: Use 'Unknown' or most frequent category
    6. COUNT FEATURES: Use 0 (no activity = 0 count)

    DOMAIN RATIONALE:
    ================
    - Physical attributes: Statistical imputation preserves distributions
    - Activity counts: 0 is meaningful (no recorded activity)
    - Age: Median imputation maintains central tendency
    - Coordinates: Missing coordinates can't be meaningfully imputed

    Args:
        df: Integrated DataFrame with missing values

    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()
    print(f"Applying missing value strategy to {len(df)} rows...")

    # 1. Building physical attributes - median imputation
    physical_cols = ['lotarea', 'bldgarea', 'comarea', 'resarea', 'officearea',
                    'retailarea', 'numfloors', 'lotfront', 'lotdepth',
                    'bldgfront', 'bldgdepth']

    for col in physical_cols:
        if col in df.columns:
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed {missing_count} missing values in {col} with median: {median_val}")

    # 2. Age features - median imputation
    if 'yearbuilt' in df.columns:
        median_year = df['yearbuilt'].median()
        missing_years = df['yearbuilt'].isnull().sum()
        if missing_years > 0:
            df['yearbuilt'] = df['yearbuilt'].fillna(median_year)
            print(f"  Imputed {missing_years} missing yearbuilt values with median: {median_year}")

        # Recalculate building age if it exists
        if 'building_age' in df.columns:
            current_year = 2025
            df['building_age'] = current_year - df['yearbuilt']

    # 3. Financial/valuation features - median imputation
    financial_cols = ['assessland', 'assesstot', 'exempttot']
    for col in financial_cols:
        if col in df.columns:
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed {missing_count} missing values in {col} with median: {median_val}")

    # 4. Activity counts - 0 imputation (no activity)
    activity_cols = ['total_transaction_count', 'transaction_count_last_3y', 'transaction_count_last_1y',
                    'has_foreclosure_activity', 'transaction_distress_score',
                    'business_count_zip', 'total_permit_count', 'active_permit_count',
                    'renovation_permit_count', 'new_building_permit_count',
                    'recent_permit_activity', 'high_construction_area',
                    'nearby_stations_count', 'high_ridership_area']

    for col in activity_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(0)
                print(f"  Imputed {missing_count} missing values in {col} with 0 (no activity)")

    # 5. Ridership features - borough-based imputation
    ridership_cols = ['avg_daily_ridership_500m', 'total_weekly_ridership_nearby']
    for col in ridership_cols:
        if col in df.columns:
            # Use borough medians for imputation
            borough_medians = df.groupby('borough')[col].median()
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df.apply(lambda row: borough_medians[row['borough']]
                                 if pd.isnull(row[col]) and row['borough'] in borough_medians.index
                                 else row[col], axis=1)
                # Fill any remaining with global median
                global_median = df[col].median()
                df[col] = df[col].fillna(global_median)
                print(f"  Imputed {missing_count} missing values in {col} with borough/global median")

    # 6. Days since last transaction - large number (no recent activity)
    if 'days_since_last_transaction' in df.columns:
        missing_count = df['days_since_last_transaction'].isnull().sum()
        if missing_count > 0:
            df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(9999)
            print(f"  Imputed {missing_count} missing values in days_since_last_transaction with 9999")

    # 7. Categorical features - mode imputation or 'Unknown'
    categorical_cols = ['bldgclass', 'landuse', 'ownername']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    print(f"  Imputed {missing_count} missing values in {col} with mode: {mode_val[0]}")
                else:
                    df[col] = df[col].fillna('Unknown')
                    print(f"  Imputed {missing_count} missing values in {col} with 'Unknown'")

    # Final check
    remaining_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = remaining_missing / total_cells * 100

    print(f"Missing value imputation complete. Remaining missing: {remaining_missing} cells ({missing_pct:.2f}%)")

    return df


def merge_dob_permits_to_pluto(pluto_df: pd.DataFrame, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Merge DOB permit data with PLUTO to add construction activity indicators.

    CONSTRUCTION ACTIVITY FEATURES:
    ==============================
    - recent_permit_count: Number of permits issued in last 2 years
    - active_construction_count: Number of permits with active status
    - renovation_permit_count: Number of alteration/renovation permits
    - new_building_permit_count: Number of new building construction permits
    - construction_value_total: Total estimated construction value
    - avg_permit_processing_days: Average time to issue permits

    BUSINESS LOGIC:
    ==============
    High construction activity can indicate:
    - Economic vitality and investment interest
    - Building upgrades and modernization
    - Potential disruption during construction
    - Future supply of new office space

    Args:
        pluto_df: PLUTO DataFrame
        data_dir: Directory containing raw data files

    Returns:
        DataFrame with DOB permit features added
    """
    try:
        dob_df = load_dob_permits(data_dir)

        # Create BBL for DOB data
        dob_df['bbl'] = dob_df['BOROUGH'].astype(str) + \
                        dob_df['Block'].astype(str).str.zfill(5) + \
                        dob_df['Lot'].astype(str).str.zfill(4)

        # Aggregate permit activity by BBL
        permit_features = dob_df.groupby('bbl').agg({
            'Job #': 'count',  # Total permits
            'Permit Status': lambda x: (x == 'ISSUED').sum(),  # Active permits
            'Job Type': lambda x: (x == 'A3').sum(),  # Renovation permits (A3 = Alteration Type 3)
            'Work Type': lambda x: (x == 'OT').sum(),  # New building permits (OT = New Building)
        }).reset_index()

        # Rename columns for clarity
        permit_features.columns = ['bbl', 'total_permit_count', 'active_permit_count',
                                 'renovation_permit_count', 'new_building_permit_count']

        # Create derived features
        permit_features['recent_permit_activity'] = (permit_features['total_permit_count'] > 0).astype(int)
        permit_features['high_construction_area'] = (permit_features['total_permit_count'] > 5).astype(int)

        print(f"DOB permit features created for {len(permit_features)} unique BBLs")

        # Merge with PLUTO
        merged_df = pluto_df.merge(permit_features, on='bbl', how='left', suffixes=('', '_dob'))

        # Fill missing values
        permit_cols = ['total_permit_count', 'active_permit_count', 'renovation_permit_count',
                      'new_building_permit_count', 'recent_permit_activity', 'high_construction_area']
        merged_df[permit_cols] = merged_df[permit_cols].fillna(0)

        print(f"Merged DOB permit features: {len(merged_df)} rows after merge")
        print(f"Buildings with recent permits: {merged_df['recent_permit_activity'].sum()}")

        return merged_df

    except FileNotFoundError:
        print("DOB permits data not found, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['total_permit_count'] = 0
        pluto_df['active_permit_count'] = 0
        pluto_df['renovation_permit_count'] = 0
        pluto_df['new_building_permit_count'] = 0
        pluto_df['recent_permit_activity'] = 0
        pluto_df['high_construction_area'] = 0
        return pluto_df
    except Exception as e:
        print(f"Error in DOB merge: {e}, adding placeholder columns")
        pluto_df = pluto_df.copy()
        pluto_df['total_permit_count'] = 0
        pluto_df['active_permit_count'] = 0
        pluto_df['renovation_permit_count'] = 0
        pluto_df['new_building_permit_count'] = 0
        pluto_df['recent_permit_activity'] = 0
        pluto_df['high_construction_area'] = 0
        return pluto_df


def create_integrated_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Create the integrated dataset by merging all available data sources.

    INTEGRATION PIPELINE OVERVIEW:
    =============================
    This function implements the complete data integration strategy following
    the BBL-centric approach outlined in the project methodology.

    PIPELINE STEPS:
    ==============
    1. START WITH PLUTO: Load the universe of NYC buildings
    2. ADD GEOCODING: Convert addresses to coordinates (for geospatial features)
    3. MERGE ACRIS: Add financial transaction history and distress signals
    4. MERGE MTA: Add transportation demand indicators (geospatial)
    5. MERGE BUSINESS: Add economic activity indicators
    6. MERGE TAX DATA: Add valuation and financial stress metrics

    QUALITY ASSURANCE:
    =================
    - All merges use LEFT JOINS to preserve PLUTO universe
    - Missing data from secondary sources is handled gracefully
    - Progress is logged at each step for transparency
    - Error handling prevents pipeline failure

    OUTPUT:
    ======
    A comprehensive dataset where each row represents one building with
    features from all available data sources, ready for feature engineering.

    Args:
        data_dir: Directory containing raw data files

    Returns:
        Integrated DataFrame ready for feature engineering
    """
    print("Creating integrated dataset...")

    # Start with PLUTO
    integrated_df = load_pluto_data(data_dir)

    # Add geocoding for geospatial features (placeholder)
    integrated_df = geocode_pluto_addresses(integrated_df)

    # Merge ACRIS (using chunked processing for memory efficiency)
    integrated_df = merge_acris_to_pluto_chunked(integrated_df, data_dir)

    # Merge MTA ridership (geospatial)
    integrated_df = merge_mta_ridership_geospatial(integrated_df, data_dir)

    # Merge business data
    integrated_df = merge_business_to_pluto(integrated_df, data_dir)

    # Merge DOB permits
    integrated_df = merge_dob_permits_to_pluto(integrated_df, data_dir)

    # Apply missing value imputation
    integrated_df = apply_missing_value_strategy(integrated_df)

    # SAVE PROCESSED DATASET
    processed_dir = Path(data_dir).parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    processed_file = processed_dir / "integrated_buildings_20251002.csv"

    print(f"Saving processed dataset to {processed_file}...")
    integrated_df.to_csv(processed_file, index=False)
    print(f"Processed dataset saved: {len(integrated_df)} rows, {len(integrated_df.columns)} columns")

    print(f"Integrated dataset created: {len(integrated_df)} rows, {len(integrated_df.columns)} columns")
    return integrated_df


if __name__ == "__main__":
    # DATA LOADING DEMONSTRATION
    # ==========================
    # This example shows how to load and inspect the PLUTO dataset
    # In a real workflow, you would call create_integrated_dataset() instead

    try:
        pluto_df = load_pluto_data()
        print("PLUTO data sample:")
        print(pluto_df.head())
        print("\nData types:")
        print(pluto_df.dtypes)

        # Additional validation checks
        print(f"\nDataset validation:")
        print(f"- Total buildings: {len(pluto_df)}")
        print(f"- Office buildings: {pluto_df['bldgclass'].str.startswith('O').sum()}")
        print(f"- Average building age: {2025 - pluto_df['yearbuilt'].mean():.1f} years")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data files are downloaded to data/raw/")