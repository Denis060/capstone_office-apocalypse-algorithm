# NYC Office Vacancy Prediction - Data Sources Documentation

**Project**: Office Apocalypse Algorithm: NYC Office Building Vacancy Risk Assessment  
**Author**: Data Science Capstone Team  
**Date**: October 2025  
**Course**: Master's Data Science Capstone Project  

## Overview

This document provides comprehensive information about all 6 datasets used in the Office Apocalypse Algorithm project. Each dataset contributes unique insights to office building vacancy risk prediction through sophisticated BBL-based integration methodology.

**Integration Summary**: 6 datasets → 7,191 office buildings → 139 engineered features → 99.99% ROC-AUC model performance

---

## 📊 Dataset Integration Architecture

```
1. PLUTO (Foundation) ──┐
2. ACRIS (Financial) ───┤
3. MTA (Accessibility) ─┼─→ BBL Integration ─→ Feature Engineering ─→ ML Model
4. Business (Economic) ─┤
5. DOB (Investment) ────┤  
6. Storefronts (Decline) ┘
```

---

## 1. Core Datasets (All 6 Required for Model Performance)

### 1.1 PLUTO (Primary Land Use Tax Lot Output)
**Source**: NYC Department of City Planning
**Purpose**: Master property dataset with comprehensive NYC real estate information
**Update Frequency**: Annual
**File Size**: ~375MB
**Rows**: 857,736
**BBL Coverage**: 100%

#### Download
- **Official Source**: [NYC Planning Open Data](https://www.nyc.gov/site/planning/data-maps/open-data.page)
- **Direct Download**: Search for "PLUTO" and download the latest version (2025 v2.1)
- **Alternative**: [Bytes of the Big Apple](https://www.bytesofthebigapple.org/)

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `BBL` | String | Borough-Block-Lot identifier | Primary key for all joins |
| `Borough` | String | Borough name/code | Used for geographic analysis |
| `Block` | Integer | Tax block number | Component of BBL |
| `Lot` | Integer | Tax lot number | Component of BBL |
| `ZipCode` | Integer | Property ZIP code | Geographic feature |
| `Address` | String | Street address | For geocoding if needed |
| `BldgClass` | String | Building classification | Office vs residential indicator |
| `LandUse` | String | Land use category | Commercial/office identification |
| `NumFloors` | Float | Number of floors | Building size indicator |
| `YearBuilt` | Integer | Construction year | Property age feature |
| `LotArea` | Integer | Lot area in sq ft | Property size |
| `BldgArea` | Integer | Building area in sq ft | Total floor area |
| `ComArea` | Integer | Commercial area in sq ft | Office space area |
| `OfficeArea` | Integer | Office area in sq ft | Target office space |
| `AssessLand` | Float | Land assessed value | Property valuation |
| `AssessTot` | Float | Total assessed value | Property valuation |

#### Data Quality Notes
- Complete BBL coverage for all NYC properties
- Some missing values in area measurements for older properties
- Building class codes need mapping to human-readable categories

---

### 1.2 ACRIS (Automated City Register Information System)
**Source**: NYC Department of Finance
**Purpose**: Real property transaction history for sales analysis
**Update Frequency**: Daily
**File Size**: ~530MB
**Rows**: 1,244,069
**BBL Coverage**: 100%

#### Download
- **Official Source**: [NYC Open Data - ACRIS](https://data.cityofnewyork.us/Housing-Development/ACRIS-Real-Property-Master/b2iz-pps8)
- **Google Drive**: [Direct Download Link](https://drive.google.com/file/d/1B73K15zu3-OmoG_qy7XGhY1i-uq7HBeQ/view?usp=drive_link)
- **Alternative**: [Bytes of the Big Apple](https://www.bytesofthebigapple.org/)

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `BBL` | String | Borough-Block-Lot identifier | Join key to PLUTO |
| `DOCUMENT ID` | String | Unique document identifier | Transaction tracking |
| `GOOD THROUGH DATE` | Date | Recording date | Transaction timestamp |
| `DOCUMENT TYPE` | String | Type of transaction | Sale, mortgage, etc. |
| `PARTY1TYPE` | String | First party type | Buyer/seller classification |
| `PARTY2TYPE` | String | Second party type | Buyer/seller classification |

#### Data Quality Notes
- Complete transaction history since 1966
- BBL format is consistent and reliable
- Some records may have missing document amounts
- Need to filter for actual sales transactions only

---

### 1.3 Vacant Storefronts
**Source**: NYC Department of Housing and Urban Development
**Purpose**: Current vacancy status for commercial properties
**Update Frequency**: Monthly
**File Size**: ~85MB
**Rows**: 348,297
**BBL Coverage**: 100%

#### Download
- **Official Source**: [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Storefronts-Reported-Vacant-or-Not-2023-/92iy-9m7x)
- **Direct Link**: Search for "Storefronts Reported Vacant" on NYC Open Data

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `BBL` | String | Borough-Block-Lot identifier | Join key to PLUTO |
| `Vacant Lot` | String | Vacancy status | Target variable |
| `Vacant On` | Date | Date vacancy reported | Duration calculation |
| `Address` | String | Property address | Verification |
| `Borough` | String | Borough name | Geographic validation |

#### Data Quality Notes
- Excellent BBL coverage and format consistency
- Clear vacancy status indicators
- Timestamps for duration calculations
- May include some duplicate reports

---

## 2. Supplementary Datasets

### 2.1 Business Registry
**Source**: NYC Department of Consumer Affairs
**Purpose**: Business activity and density analysis
**Update Frequency**: Weekly
**File Size**: ~16MB
**Rows**: 66,425
**BBL Coverage**: 63% (41,716 valid BBLs)

#### Download
- **Official Source**: [NYC Open Data - Business Licenses](https://data.cityofnewyork.us/Business/License-Applications/ptev-4hud)
- **Alternative**: Search for "Business Licenses" on NYC Open Data

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `License Number` | String | Business license ID | Unique identifier |
| `Business Name` | String | Legal business name | Business identification |
| `Address` | String | Business address | Geocoding required |
| `Borough` | String | Borough location | Geographic feature |
| `License Status` | String | Active/Inactive | Business vitality |
| `License Type` | String | Business category | Industry classification |

#### Data Quality Notes
- 37% missing BBLs require geocoding
- Address data quality varies
- License status needs cleaning
- Good coverage of active businesses

---

### 2.2 MTA Subway Ridership
**Source**: Metropolitan Transportation Authority
**Purpose**: Foot traffic and accessibility analysis
**Update Frequency**: Daily
**File Size**: ~16MB
**Rows**: 769,148
**BBL Coverage**: 0% (station-level data)

#### Download
- **Official Source**: [MTA Open Data](https://new.mta.info/developers/download-agreement)
- **Direct Link**: [MTA Ridership Data](https://new.mta.info/agency/new-york-city-transit/subway-bus-ridership-2024)

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `transit_timestamp` | DateTime | Date and time of measurement | Temporal aggregation |
| `station_complex_id` | String | Station identifier | Geographic reference |
| `ridership` | Integer | Entry/exit count | Foot traffic measure |
| `latitude` | Float | Station latitude | Spatial join coordinate |
| `longitude` | Float | Station longitude | Spatial join coordinate |

#### Data Quality Notes
- High-quality ridership counts
- Station coordinates are accurate
- Requires spatial distance calculations
- Good temporal coverage (2020-2024)

---

### 2.3 DOB Building Permits
**Source**: NYC Department of Buildings
**Purpose**: Construction activity and renovation analysis
**Update Frequency**: Daily
**File Size**: ~1.45GB
**Rows**: 4,000,000+
**BBL Coverage**: TBD

#### Download
- **Official Source**: [NYC Open Data - DOB Permits](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a)
- **Direct Link**: Search for "DOB Permit Issuance" on NYC Open Data

#### Key Columns
| Column | Type | Description | Integration Notes |
|--------|------|-------------|-------------------|
| `BBL` | String | Borough-Block-Lot identifier | Join key to PLUTO |
| `Permit Type` | String | Type of permit | Construction activity |
| `Work Type` | String | Nature of work | Renovation vs new construction |
| `Filing Date` | Date | Permit filing date | Temporal analysis |
| `Expiration Date` | Date | Permit expiration | Activity duration |
| `Estimated Cost` | Float | Project cost estimate | Investment indicator |

#### Data Quality Notes
- Very large dataset requires chunked processing
- BBL coverage needs validation
- Cost estimates may be missing for some permits
- Rich temporal data for activity analysis

---

## 3. Data Integration Strategy

### 3.1 Primary Integration Method: BBL Joins
Most datasets can be joined directly using the BBL as the primary key:

```python
# Core integration example
master_df = pluto_df.merge(
    acris_aggregated,
    on='bbl',
    how='left'
).merge(
    vacant_storefronts,
    on='bbl',
    how='left'
)
```

### 3.2 Spatial Integration for Missing BBLs
For datasets without BBLs, use coordinate-based spatial joins:

```python
import geopandas as gpd
from shapely.geometry import Point

# Convert to GeoDataFrames
properties_gdf = gpd.GeoDataFrame(
    pluto_df,
    geometry=gpd.points_from_xy(pluto_df.longitude, pluto_df.latitude)
)

stations_gdf = gpd.GeoDataFrame(
    mta_df,
    geometry=gpd.points_from_xy(mta_df.longitude, mta_df.latitude)
)

# Spatial join with distance threshold
joined = gpd.sjoin_nearest(
    properties_gdf,
    stations_gdf,
    distance_col='station_distance',
    max_distance=1000  # 1km threshold
)
```

### 3.3 Data Aggregation Strategy

#### MTA Ridership Aggregation
```python
# Aggregate to monthly station level
monthly_ridership = mta_df.groupby([
    mta_df['transit_timestamp'].dt.to_period('M'),
    'station_complex_id'
])['ridership'].sum().reset_index()
```

#### ACRIS Transaction Aggregation
```python
# Create property-level features
property_features = acris_df.groupby('bbl').agg({
    'good_through_date': 'max',  # Last transaction
    'document_id': 'count',      # Transaction frequency
    'sale_price': ['mean', 'max']  # Price statistics
}).reset_index()
```

---

## 4. Data Quality Validation

### 4.1 BBL Format Validation
```python
def validate_bbl_format(bbl_series):
    \"\"\"Validate BBL format (10 digits)\"\"\"
    return bbl_series.astype(str).str.match(r'^\d{10}$')

# Usage
valid_bbls = validate_bbl_format(df['bbl'])
print(f"Valid BBLs: {valid_bbls.sum()}/{len(df)}")
```

### 4.2 Coordinate Validation
```python
def validate_nyc_coordinates(lat_series, lon_series):
    \"\"\"Validate coordinates are within NYC bounds\"\"\"
    nyc_bounds = {
        'lat_min': 40.4774, 'lat_max': 40.9176,
        'lon_min': -74.2591, 'lon_max': -73.7004
    }

    valid_lat = (lat_series >= nyc_bounds['lat_min']) & (lat_series <= nyc_bounds['lat_max'])
    valid_lon = (lon_series >= nyc_bounds['lon_min']) & (lon_series <= nyc_bounds['lon_max'])

    return valid_lat & valid_lon
```

---

## 5. Memory Optimization Strategies

### 5.1 Chunked Processing for Large Files
```python
def process_large_file_in_chunks(file_path, chunk_size=500000):
    \"\"\"Process large CSV files in chunks\"\"\"
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)

    return pd.concat(chunks, ignore_index=True)
```

### 5.2 Memory-Efficient Data Types
```python
# Optimize dtypes for PLUTO
pluto_dtypes = {
    'BBL': str,           # Keep as string for leading zeros
    'BoroCode': 'Int8',   # Small integer range
    'Block': 'Int32',     # Reasonable range
    'Lot': 'Int16',       # Small range
    'ZipCode': 'Int32',   # Standard ZIP
    'LandUse': 'category', # Limited categories
    'YearBuilt': 'Int16',  # Year range
    'NumFloors': 'float32', # Reduce precision
    'AssessLand': 'float32' # Reduce precision
}

df = pd.read_csv('pluto.csv', dtype=pluto_dtypes)
```

---

## 6. Update Schedule & Version Control

### 6.1 Dataset Update Frequencies
- **PLUTO**: Annual (typically March)
- **ACRIS**: Daily updates
- **Vacant Storefronts**: Monthly
- **Business Registry**: Weekly
- **MTA Ridership**: Daily (with monthly summaries)
- **DOB Permits**: Daily

### 6.2 Version Control Strategy
```gitignore
# Large raw data files
data/raw/*.csv

# Processed datasets
data/processed/*.csv

# Keep small reference files
!data/raw/README.md
!data/processed/README.md
```

### 6.3 Data Refresh Process
1. Download latest datasets from official sources
2. Update file paths in notebooks
3. Re-run data acquisition pipeline
4. Validate BBL coverage and data quality
5. Update integration scripts if needed

---

## 7. Troubleshooting Common Issues

### 7.1 Memory Errors
**Symptom**: `MemoryError` when loading large files
**Solution**: Use chunked processing or increase system memory

### 7.2 BBL Format Issues
**Symptom**: BBLs not matching between datasets
**Solution**: Standardize BBL format to 10 digits with leading zeros

### 7.3 Coordinate System Mismatches
**Symptom**: Spatial joins failing
**Solution**: Ensure consistent coordinate reference system (EPSG:4326 for lat/lon)

### 7.4 Missing Data
**Symptom**: High percentage of null values after joins
**Solution**: Validate data sources and consider alternative integration methods

---

## 8. Contact & Support

For questions about datasets or integration:
- **NYC Open Data**: [Support Portal](https://data.cityofnewyork.us/)
- **MTA Open Data**: [Developer Resources](https://new.mta.info/developers)
- **Project Issues**: [GitHub Issues](https://github.com/Denis060/office-apocalypse-algorithm/issues)

---

*Last updated: September 16, 2025*
